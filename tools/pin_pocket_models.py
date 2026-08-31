"""Resolve and pin the PocketTTS Windows CPU closure, model and official voices.

Not part of the extension and never imported by it. A maintainer's tool, run on
a machine that can reach pypi.org and huggingface.co, which is why it lives in
``tools/`` rather than in ``scripts/`` where Forge would import it at start-up.
It is the third of the pinners, after ``tools/pin_voice_models.py`` and
``tools/pin_sopro_models.py``, and it is the one engine whose manifest needs
both halves of the job in a single run: a wheel closure PyPI serves digests for,
and model artifacts that live on the hub in two repositories, one of them behind
an access gate.

Why a tool rather than a hand-written manifest
----------------------------------------------
PocketTTS declares dependency floors, and a floor is not a runtime identity.
Two machines that both satisfy ``torch>=2.4`` can be running different attention
kernels, different thread pools and different serialization behaviour, and for
this engine that difference is not academic: a saved voice state is a tensor
written by one build and read back by another, so "the same PocketTTS" has to
mean the same bytes rather than the same version string. This repository
therefore pins exact builds and the fingerprint of every byte, and this tool is
how that list is produced without anybody transcribing fifty-two SHA-256 digests
by hand.

    python tools/pin_pocket_models.py --check      # report, change nothing
    python tools/pin_pocket_models.py              # rewrite the closure
    python tools/pin_pocket_models.py --model      # also resolve the model and its voices

Exit codes: 0 when there is nothing left to do, 1 when something is still
unresolved -- a ``--check`` with work outstanding, or a run that wrote the
public half and left the gated one behind its gate -- and 2 when a maintainer
has to look at something before running it again.

Why the closure is read from the manifest rather than written here
------------------------------------------------------------------
``tools/pin_sopro_models.py`` keeps its ``CLOSURE`` in the tool. Pocket's lives
in ``runtime.closure`` in the manifest, as ``[package, version, kind]`` triples,
and this tool reads it from there. The list is still written down, in full, by a
person who read PocketTTS's imports -- that is the property this whole design
exists to protect and it has not moved -- but the reviewed artifact is the
manifest, because the manifest is the file somebody opens when they want to know
what a machine will be asked to install. A closure that lived in the tool would
be a closure that is invisible in the diff that changes it.

What it will not do
-------------------
It will not resolve a dependency graph. "Whatever pip decides today" is the
thing being removed, so this tool turns a written-down list into sizes and
hashes and fails loudly when PyPI no longer serves a wheel that list names.

It will not decide for itself which files a model consists of, or which voices
are official. :data:`MODEL_FILES`, :data:`CLONING_FILES` and :data:`VOICES` are
tables a person filled in after reading the model card; the hub is asked what
those files *are*, never what they *should be*. Section 10 is explicit that the
official voice list must not be scraped from whatever upstream currently ships,
and a state the repository serves that is not on the list is reported rather
than adopted.

It will not add a Linux platform. PyPI's Linux ``torch`` wheels pull the whole
CUDA closure through their dependency markers, and the first Pocket release
neither claims nor consumes a graphics device. A Linux CPU closure comes from
PyTorch's own ``+cpu`` index and is a separate, deliberate addition with its own
gate -- not something a script guesses at because it happened to be run on
Linux.

It will not change a digest that is already checked in. The committed hash is
the trust root for the whole feature, so a disagreement between the manifest and
what a publisher serves today fails the entire run and prints both. A publisher
who has re-uploaded is a review decision -- somebody has to read the new model
card and decide -- and not something a script gets to make at three in the
morning.

It will not touch ``voice/managed-voice-models.json`` or
``voice/managed-sopro-models.json``. Kokoro's and Sopro's closures are different
trust roots with different lifecycles, and a tool that could edit all three is a
tool that can break two working engines while installing the third. That refusal
is in :func:`main` as well as in this paragraph.

It will not accept anybody's licence for them, and it does not write a
credential anywhere. If ``HF_TOKEN`` is in this process's environment it is used
to ask the hub about the gated repository and is then dropped; nothing derived
from it reaches the manifest, and the manifest it writes is a public document.

The gated half is a state, not a failure
----------------------------------------
``kyutai/pocket-tts`` answers 401 or 403 until a person has accepted Kyutai's
conditions with their own account. That is section 23.3's two-state world --
"Pocket can speak" and "Pocket can clone" are separate facts -- so a closed gate
here prints a sentence, leaves the public half written, and exits 1. It is not
an error, and a run that treated it as one would be a run that threw away four
resolved artifacts because a fifth needed a licence.

Where it writes
---------------
Into the tracked manifest, in place, the way the Sopro pinner does: PyPI and the
hub both serve digests, so nothing is downloaded and there is no local
measurement whose result would belong to one machine. It never writes
``voice/managed-pocket-models.local.json``. That file is written by
:mod:`mc_voice_models` at install time, holds the digests recorded for artifacts
nobody pinned, is untracked for the reason the Kokoro and Sopro overlays are
untracked, and is not this tool's business.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "voice" / "managed-pocket-models.json"

SIBLINGS = ("managed-voice-models.json", "managed-sopro-models.json",
            "managed-cleanup-models.json")
"""The manifests this tool refuses to open, by name.

The docstring above promises it; this is the promise in code. ``--manifest``
exists so a maintainer can pin a copy before committing it, and a copy is not
the same thing as a sibling engine's trust root.
"""

SCHEMA = 1
"""The manifest layout this tool writes, and the one ``mc_voice_pocket`` reads."""

TIMEOUT = 120.0
USER_AGENT = "ModelChain-Pocket-Pinner"

PYTHONS = ("3.10", "3.11", "3.12", "3.13")
"""The Python minors this release advertises for PocketTTS on Windows.

An allowlist rather than a range, and section 23.1's point exactly: upstream
declaring 3.10 through 3.14 establishes feasibility, and a *combination* becomes
supported only after the exact closure has been installed and self-tested on
that class. Adding 3.14 here is a release decision, not a version bump, and the
manifest's platform list is checked against this tuple rather than generated
from it -- so a platform somebody added by hand fails loudly instead of being
silently pinned.
"""

PURE = "pure"
BINARY = "binary"
ABI3 = "abi3"
WINDOWS = "windows"
"""How to recognise the one wheel that belongs to a platform.

``PURE``   -- ``py3-none-any`` (or ``py2.py3-none-any``), one file for everything.
``BINARY`` -- ``cp310-cp310-win_amd64``, one file per Python minor.
``ABI3``   -- ``cp310-abi3-win_amd64``, built once against the stable ABI and
              valid for every later minor, which is what safetensors ships.
``WINDOWS``-- ``py3-none-win_amd64``, a platform wheel with no CPython tag.

Four kinds rather than the three the closure uses today, because the closure is
data in the manifest: a maintainer who adds ``soundfile`` to ``runtime.closure``
must not have to edit this file to describe how its wheel is named.
"""

KINDS = (PURE, BINARY, ABI3, WINDOWS)

MODEL_ID = "english"
"""The model this tool resolves, and the only one the first release ships.

Named rather than "whatever is in the manifest", because resolving every entry
of a catalogue that may later hold a second language is a decision somebody
should take with ``--model`` in front of them, not one this loop takes for them.
"""

API = "https://huggingface.co/api/models/{repo}/revision/{revision}"
RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
BLOB = "https://huggingface.co/{repo}/blob/{revision}/{path}"

CREDENTIAL_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
"""Where a Hugging Face token is read from, and the same three names
:mod:`mc_voice_models` reads. Read fresh, never cached, never written."""

HOSTS = ("huggingface.co", "www.huggingface.co")
"""The only hosts a credential is ever attached to. The hub redirects delivery
to a signed URL on a CDN, and a token that followed it there would be a token
handed to a third party for no reason (I-PKT-21)."""

SHA256_HEADERS = ("x-linked-etag", "x-checksum-sha256", "x-amz-meta-sha256")
"""Headers whose *name* asserts that the value is a SHA-256.

A plain ``ETag`` is believed only when it says ``sha256:`` of itself. A bare
64-hex ``ETag`` is an opaque validator under RFC 9110 -- it looks exactly like a
digest and is not one -- and reading it as a digest is how a delivery host's
cache key ends up committed as a model's identity.
"""

STATE_SUFFIX = ".safetensors"
"""What an installed official voice state is called on disk.

``mc_voice_pocket._official_voices_ready`` looks for
``official/<model-id>/<voice-id>.safetensors``, so a voice's ``local_name`` is
its id and this suffix, whatever the repository calls the file. The two are
allowed to differ and do: upstream keeps its states under ``voices/`` and the
installer promotes a directory rather than a tree.
"""


@dataclass(frozen=True)
class Declared:
    """One file a model bundle is made of, as a person wrote it down.

    ``path`` is what the repository serves and ``local_name`` is what the
    installed directory calls it. They are separate fields because the gated
    weights are also called ``model.safetensors`` upstream, and installing them
    under that name would write the cloning-capable model over the public one --
    which is precisely the file ``weights_path_without_voice_cloning`` still has
    to name after the gated half arrives.
    """

    path: str
    local_name: str
    about: str
    config_key: str = ""


MODEL_FILES = (
    Declared("languages/english/model.safetensors", "model.safetensors",
             "the transformer, the depth decoder and the acoustic decoder",
             "weights_path_without_voice_cloning"),
    Declared("languages/english/tokenizer.model", "tokenizer.model",
             "the sentencepiece text tokenizer", "tokenizer_path"),
)
"""What ``kyutai/pocket-tts-without-voice-cloning`` has to serve.

Written down rather than discovered, for the reason the Sopro pinner gives about
``sopro.hub.ARTIFACT_FILES``: the machine writing the manifest is not the
machine that has the runtime, so it cannot import PocketTTS to ask, and a list
built from "everything in the repository" would install a model card and a
sample WAV and call them required. The tool refuses if a name here is not served
and prints what the repository actually holds, so a rename upstream is a loud
failure with the correction already on screen.

The paths and the revisions are upstream's own, read out of
``pocket_tts/config/english.yaml`` inside the 3.0.2 wheel: that file is what the
package resolves when nobody replaces its locations, so it is the statement of
which bytes this model *is*. There is no ``config.json`` here because there is
nothing for one to do -- the architecture is that same shipped YAML, and the
installer copies it out of the runtime rather than fetching a second description
of the model from the hub (see ``mc_voice_pocket._read_recipe``).
"""

CLONING_FILES = (
    Declared("languages/english/model.safetensors", "model-voice-cloning.safetensors",
             "the cloning-capable weights, which carry the voice encoder the "
             "public model does not", "weights_path"),
)
"""What the gated ``kyutai/pocket-tts`` adds, and deliberately only that.

The gated repository serves its own copy of the tokenizer as well. It is not
listed: the public half already installed that file and verified it, and a
second copy would be a second trust root for one artifact plus a second thing
that can disagree with itself.

Note that the two repositories are pinned at *different* revisions, which is
upstream's own arrangement rather than an oversight here -- its config names one
commit for the cloning weights and another for the public weights and the
tokenizer -- so the manifest carries a revision per repository.
"""


@dataclass(frozen=True)
class Voice:
    """One official voice, as a person reviewed it."""

    identifier: str
    display_name: str
    language: str
    accent: str
    path: str


VOICES = (
    Voice("alba", "Alba", "en", "Scottish", "voices/alba.safetensors"),
    Voice("marlow", "Marlow", "en", "English (Received Pronunciation)",
          "voices/marlow.safetensors"),
    Voice("juno", "Juno", "en", "American (General American)",
          "voices/juno.safetensors"),
    Voice("rhys", "Rhys", "en", "Welsh", "voices/rhys.safetensors"),
)
"""The official voice bank, written down rather than discovered.

Section 10 and I-PKT-22: the catalogue somebody chooses from is a reviewed list
in this repository with a source, a licence and an attribution beside every
entry, not a directory listing. Adding a voice means listening to it, reading
its terms and changing this tuple in a commit somebody approved. A state the
repository serves that is not named here is reported by :func:`_extras` and left
alone, and a state named here that the repository no longer serves fails the
run -- a catalogue that offers four voices and can speak three is a catalogue
where choosing the wrong one is silence.
"""

VOICE_LICENSE = "CC-BY-4.0 (PocketTTS voice states, Kyutai)"

VOICE_ATTRIBUTION = ("PocketTTS official voice {display_name} by Kyutai (CC-BY-4.0). "
                     "Precomputed voice state from huggingface.co/{repo}, installed "
                     "locally and never fetched at speech time.")

NOTE_PINNED = (
    "The runtime closure below is resolved: every wheel is named, sized and hashed from "
    "pypi.org, so the managed install fetches exactly what this repository claims and "
    "refuses anything else. Model and voice artifacts carry the publisher's digest where "
    "the hub reports one and are attested over HTTPS at install time where it does not, "
    "and either way the digest of what actually arrived is recorded. Regenerate this file "
    "with tools/pin_pocket_models.py; it is the only thing that should write it. This is "
    "GATE P-1 satisfied: no unreviewed wheel is fetched dynamically.")

NOTE_PROVISIONAL = (
    "PROVISIONAL. The runtime closure below is written down but not yet resolved: at least "
    "one artifact list is empty or unhashed, so the managed install refuses and says why. A "
    "maintainer on a machine that can reach pypi.org and huggingface.co runs "
    "tools/pin_pocket_models.py, which fills in filenames, byte counts and SHA-256 digests "
    "from the publishers and sets \"pinned\" to true. Until then Pocket installs only from a "
    "folder somebody filled themselves, and what they supply is checked against "
    "required_paths and has its digests recorded. This is GATE P-1: no unreviewed wheel is "
    "fetched dynamically, and an artifact this repository makes no claim about is an "
    "artifact it will not download.")


class PinError(RuntimeError):
    """Something a maintainer has to look at rather than re-run."""


class Gate(RuntimeError):
    """The publisher's access gate, which is a state rather than a failure.

    Voice Chat cannot accept Kyutai's conditions on anybody's behalf, legally or
    technically, and this tool does not pretend to. Raised only for the gated
    repository, caught in :func:`build`, and reported as the sentence a
    maintainer needs -- with the public half already resolved and about to be
    written.
    """

    def __init__(self, repo: str, status: int):
        self.repo = str(repo or "")
        self.status = int(status or 0)
        super().__init__(
            f"huggingface.co/{self.repo} answered HTTP {self.status}. Accept the model's "
            f"conditions on that page with your own account, then run this again with "
            f"HF_TOKEN set in this process's environment.")


@dataclass
class State:
    """What one run reached, so that :func:`main` can say it in one place."""

    disagreements: list = field(default_factory=list)
    gate: str = ""
    pinned: bool = False
    wheels: int = 0
    files: int = 0
    voices: int = 0
    cloning: int = 0


# --------------------------------------------------------------------------- #
# PyPI
# --------------------------------------------------------------------------- #


def _release(name: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            return json.load(answer)
    except Exception as exc:
        raise PinError(f"{name} {version} could not be read from PyPI ({exc})") from None


def _wheel(entry: dict, kind: str, python: str) -> dict:
    """The one wheel from a release that belongs to ``python`` on Windows x86-64.

    Refuses ambiguity rather than picking. Two candidates means the selector
    below no longer describes what the publisher ships, and quietly taking the
    first one is how a free-threaded ``cp313t`` build ends up pinned as the
    ordinary ``cp313`` one.
    """
    tag = "cp" + python.replace(".", "")
    found = []
    for item in entry.get("urls") or ():
        if item.get("packagetype") != "bdist_wheel":
            continue
        name = str(item.get("filename") or "")
        if kind == PURE:
            if name.endswith("-none-any.whl"):
                found.append(item)
        elif kind == WINDOWS:
            if name.endswith("-none-win_amd64.whl"):
                found.append(item)
        elif kind == ABI3:
            if "-abi3-win_amd64.whl" in name:
                found.append(item)
        elif name.endswith(f"-{tag}-{tag}-win_amd64.whl"):
            found.append(item)
    if not found:
        raise PinError(f"{entry['info']['name']} {entry['info']['version']} has no "
                       f"{kind} wheel for Python {python} on Windows x86-64")
    if len(found) > 1:
        names = ", ".join(sorted(str(item.get("filename")) for item in found))
        raise PinError(f"{entry['info']['name']} {entry['info']['version']} offers more than "
                       f"one candidate for Python {python}: {names}")
    return found[0]


def _wheel_artifact(item: dict) -> dict:
    """One PyPI wheel as the five keys the manifest's artifact lists carry.

    A wheel with no digest or no size is refused rather than written unpinned.
    PyPI serves both for every file it hosts, so an answer without them is a
    broken read, and a runtime closure is the one list in this feature where
    "the publisher will tell us at install time" is not good enough.
    """
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
# The hub
# --------------------------------------------------------------------------- #


class _StopAtRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect handler that follows nothing, so :func:`_head` can decide."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _credential() -> str:
    """A Hugging Face token from this process's environment, or nothing.

    Read here, used for two requests and never stored. Nothing in this
    repository writes these variables, no message prints one, and the manifest
    this tool produces is a public document that contains no trace of one.
    """
    for name in CREDENTIAL_VARIABLES:
        found = str(os.environ.get(name) or "").strip()
        if found:
            return found
    return ""


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.casefold()


def _header_map(answer) -> dict:
    return {str(name).casefold(): value for name, value in answer.headers.items()}


def _headers(url: str, token: str) -> dict:
    """The request headers, with the credential only where it belongs."""
    found = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if token and _host(url) in HOSTS:
        found["Authorization"] = f"Bearer {token}"
    return found


def _get(url: str, token: str = "") -> tuple:
    """``(status, body)`` for one hub API read."""
    request = urllib.request.Request(url, headers=_headers(url, token))
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            status = int(getattr(answer, "status", 200) or 200)
            return status, answer.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as answer:
        with contextlib.closing(answer):
            return int(getattr(answer, "code", 0) or 0), ""
    except Exception as exc:
        raise PinError(f"huggingface.co could not be reached ({exc})") from None


def _head(url: str, token: str = "", hops: int = 5) -> tuple:
    """``(status, headers)`` for one artifact, without leaving the publisher.

    The loop stops the moment a redirect points at another host and answers with
    the last headers the publisher itself sent. That is not a workaround: the
    hub's delivery CDN returns an ``ETag`` of its own that is 64 hex characters
    and is not the file's digest, so a reader that followed the hop would pin a
    cache key as a model's identity. It is also where the credential stops --
    :func:`_headers` attaches one only for the hub, so a hop off it is a request
    with nothing to leak.
    """
    opener = urllib.request.build_opener(_StopAtRedirect)
    here = url
    status, found = 0, {}
    for _ in range(max(int(hops), 1)):
        request = urllib.request.Request(here, method="HEAD",
                                         headers=_headers(here, token))
        try:
            with contextlib.closing(opener.open(request, timeout=TIMEOUT)) as answer:
                return int(getattr(answer, "status", 200) or 200), _header_map(answer)
        except urllib.error.HTTPError as answer:
            with contextlib.closing(answer):
                status = int(getattr(answer, "code", 0) or 0)
                found = _header_map(answer)
        except Exception as exc:
            raise PinError(f"{url.split('?')[0]} could not be read from the hub "
                           f"({exc})") from None
        if status not in (301, 302, 303, 307, 308):
            return status, found
        target = str(found.get("location") or "").strip()
        if not target:
            return status, found
        target = urllib.parse.urljoin(here, target)
        if _host(target) != _host(here):
            return 200, found
        here = target
    return status or 200, found


def _as_sha256(raw, prefixed: bool) -> str:
    """One header value read as a SHA-256, or nothing.

    ``prefixed`` says whether the value has to introduce itself. See
    :data:`SHA256_HEADERS` for why a plain ``ETag`` has to.
    """
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
    """The real size of the file.

    ``x-linked-size`` first: a model on the hub is an LFS object, and a plain
    ``content-length`` on the pointer would be a few hundred bytes -- a number
    that looks like an answer and is not one.
    """
    for name in ("x-linked-size", "content-length"):
        raw = str(headers.get(name) or "")
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 0


def repository(repo: str, revision: str, token: str, say) -> tuple:
    """``(commit, served)`` -- what ``revision`` points at, and every file there.

    The commit is read once and every artifact URL below is built against it. A
    branch is a name for whatever it points at today, and a manifest that named
    one would be a manifest whose meaning changes without a commit in this
    repository.
    """
    status, body = _get(API.format(repo=repo, revision=revision), token)
    if status in (401, 403):
        raise Gate(repo, status)
    if status != 200:
        raise PinError(f"{repo}: the hub answered {status} for revision {revision}")
    try:
        answer = json.loads(body)
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


def _must_serve(repo: str, commit: str, served: tuple, path: str) -> None:
    """Refuse a file the repository does not have, and say what it does have.

    The listing is printed because the correction is almost always a rename
    upstream, and a maintainer who has to go and look it up is a maintainer who
    will run this tool three times.
    """
    if path in served:
        return
    listing = "\n  ".join(served) or "(nothing)"
    raise PinError(f"{repo} at {commit[:12]} does not serve {path}, which this tool's table "
                   f"names. Correcting the table is a reviewed change; the repository holds:"
                   f"\n  {listing}")


def _hub_artifact(repo: str, commit: str, path: str, local_name: str, token: str) -> dict:
    """One hub file as a manifest artifact, pinned as far as the hub allows.

    ``bytes`` and ``sha256`` are ``None`` when the hub reports neither, which is
    the ordinary answer for a small file kept in git rather than in LFS. That is
    a state rather than a gap: :func:`mc_voice_models._resolve` attests such an
    artifact against the publisher over HTTPS at install time and records the
    digest of what actually arrived, and writing a hash nobody read would be
    worse than writing none.
    """
    url = RESOLVE.format(repo=repo, revision=commit, path=urllib.parse.quote(path))
    status, headers = _head(url, token)
    if status in (401, 403):
        raise Gate(repo, status)
    if status >= 400:
        raise PinError(f"{repo}: the hub answered {status} for {path} at {commit[:12]}")
    return {
        "filename": path,
        "local_name": local_name,
        "url": url,
        "bytes": _published_size(headers) or None,
        "sha256": _published_sha256(headers) or None,
    }


def _describe(artifact: dict) -> str:
    size = artifact.get("bytes")
    digest = artifact.get("sha256")
    if size and digest:
        return f"{size} bytes, sha256 {digest[:12]}"
    if size:
        return f"{size} bytes, no published digest (attested at install time)"
    return "neither a size nor a digest was published (attested at install time)"


# --------------------------------------------------------------------------- #
# The closure
# --------------------------------------------------------------------------- #


def closure(existing: dict) -> tuple:
    """The ``[package, version, kind]`` triples the manifest writes down.

    Validated before anything is asked of PyPI, so a typo in a wheel kind is a
    sentence rather than a package read that fails for a reason nobody can see.
    """
    runtime = existing.get("runtime")
    if not isinstance(runtime, dict):
        raise PinError("the manifest declares no runtime")
    written = runtime.get("closure")
    if not isinstance(written, list) or not written:
        raise PinError("the manifest declares no runtime.closure, which is the list this "
                       "tool resolves. It is written by a person who read PocketTTS's "
                       "imports, and this tool does not invent one.")
    found = []
    for item in written:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise PinError(f"{item!r} in runtime.closure is not a "
                           f"[package, version, kind] triple")
        name, version, kind = (str(part).strip() for part in item)
        if not name or not version:
            raise PinError(f"{item!r} in runtime.closure names no package or no version")
        if kind not in KINDS:
            raise PinError(f"{name} {version} names the wheel kind {kind!r}, which this "
                           f"tool does not know ({', '.join(KINDS)})")
        found.append((name, version, kind))
    return tuple(found)


def _version_of(entries: tuple, package: str) -> str:
    wanted = package.replace("_", "-").casefold()
    for name, version, _kind in entries:
        if name.replace("_", "-").casefold() == wanted:
            return version
    return ""


def _self_consistent(runtime: dict, entries: tuple) -> None:
    """Refuse a manifest that disagrees with itself.

    ``runtime.version`` and ``runtime.torch_version`` are what the panel and the
    installed marker report, and the closure is what actually gets installed. A
    build where those two say different things is a build whose "Installed --
    PocketTTS 3.0.2" line is a claim about a wheel it did not fetch.
    """
    for key, package in (("version", "pocket-tts"), ("torch_version", "torch")):
        declared = str(runtime.get(key) or "")
        written = _version_of(entries, package)
        if declared and written and declared != written:
            raise PinError(f"the manifest says runtime.{key} is {declared} and its closure "
                           f"pins {package} {written}. One of the two is wrong, and this "
                           f"tool will not choose.")


def platforms(entries: tuple, declared: list, state: State, committed: dict, say) -> list:
    """One fully pinned wheel closure per platform the manifest advertises.

    Each release is read from PyPI once and then selected from four times, so a
    thirteen-package closure across four minors is thirteen requests rather than
    fifty-two.
    """
    if not declared:
        raise PinError("the manifest advertises no platforms")
    minors = tuple(sorted({str(item.get("python") or "") for item in declared}))
    if minors != tuple(sorted(PYTHONS)):
        raise PinError(f"the manifest advertises Python {', '.join(minors)} and this tool "
                       f"is written for {', '.join(sorted(PYTHONS))}. A supported "
                       f"combination is a release decision -- see PYTHONS.")
    for item in declared:
        if str(item.get("system") or "") != "windows":
            raise PinError(f"{item.get('id')} is not a Windows platform. This closure is "
                           f"Windows x86-64; a Linux CPU closure comes from PyTorch's own "
                           f"+cpu index and is a separate, deliberate addition.")

    releases = {}
    for name, version, _kind in entries:
        say(f"  reading {name} {version} from PyPI")
        releases[(name, version)] = _release(name, version)

    found = []
    for item in declared:
        python = str(item.get("python") or "")
        artifacts = []
        for name, version, kind in entries:
            artifact = _wheel_artifact(_wheel(releases[(name, version)], kind, python))
            _agree(state, committed, artifact, str(item.get("id") or python))
            artifacts.append(artifact)
        total = sum(int(one["bytes"]) for one in artifacts)
        say(f"  {item.get('id')}: {len(artifacts)} wheels, {total / 1e6:.0f} MB")
        state.wheels += len(artifacts)
        rebuilt = dict(item)
        rebuilt["artifacts"] = artifacts
        found.append(rebuilt)
    return found


def _complete(declared: list) -> bool:
    """Whether every platform has a closure that could actually be installed.

    The same question :func:`mc_voice_pocket.pinned` asks of the same data, and
    the reason the top-level ``pinned`` flag can be trusted: it is written from
    the artifacts rather than by a person who believed the run had finished.
    """
    if not declared:
        return False
    for item in declared:
        artifacts = item.get("artifacts") or ()
        if not artifacts:
            return False
        for one in artifacts:
            if len(str(one.get("sha256") or "")) != 64 or int(one.get("bytes") or 0) <= 0:
                return False
    return True


# --------------------------------------------------------------------------- #
# The model, its voices and the gated half
# --------------------------------------------------------------------------- #


def _committed(existing: dict) -> dict:
    """Every digest already checked in, by filename.

    Collected before anything is resolved so that :func:`_agree` can compare
    against the manifest as it stands rather than against the half-rewritten one
    this run is building.

    One filename means one digest. A pure wheel appears in all four platform
    closures and a later entry must never quietly replace an earlier one: a map
    that let it would be a map where a re-uploaded ``py3-none-any`` wheel is
    compared against itself and passes. Two digests for one name is a manifest
    contradicting itself, which is a refusal rather than a merge.
    """
    found = {}

    def keep(entry):
        if not (isinstance(entry, dict) and entry.get("filename") and entry.get("sha256")):
            return
        name, digest = str(entry["filename"]), str(entry["sha256"]).casefold()
        was = found.setdefault(name, digest)
        if was != digest:
            raise PinError(f"the manifest gives {name} two digests, {was} and {digest}. "
                           f"One artifact has one identity, and this tool will not choose "
                           f"which of the two it is.")

    for platform in ((existing.get("runtime") or {}).get("platforms") or ()):
        for item in platform.get("artifacts") or ():
            keep(item)
    for entry in (existing.get("models") or {}).values():
        if not isinstance(entry, dict):
            continue
        for key in ("files", "cloning_files"):
            for item in entry.get(key) or ():
                keep(item)
        for voice in entry.get("voices") or ():
            if isinstance(voice, dict):
                keep(voice.get("artifact"))
    return found


def _agree(state: State, committed: dict, artifact: dict, where: str) -> None:
    """Record, never resolve, a disagreement with what is checked in.

    Collected rather than raised so that one run names every changed artifact
    instead of the first one. The whole run then writes nothing: see
    :func:`main`.
    """
    name = str(artifact.get("filename") or "")
    fresh = str(artifact.get("sha256") or "").casefold()
    was = committed.get(name, "")
    if was and fresh and was != fresh:
        state.disagreements.append(f"{where}/{name}: checked in {was}, published {fresh}")


def _extras(repo: str, served: tuple, known: set, say) -> None:
    """Say what the repository holds that this build does not install.

    Reported and not adopted. A voice state appearing upstream is upstream's
    news; it becomes a voice this extension offers when somebody has listened to
    it and written it into :data:`VOICES`. Everything already declared -- the
    model's own tensors included -- is left out of the count, because a line
    that called ``model.safetensors`` an unreviewed voice would be a line
    nobody reads twice.
    """
    extra = [name for name in served
             if name.endswith(STATE_SUFFIX) and name not in known]
    if extra:
        say(f"  {repo} also serves {len(extra)} state(s) this build does not install: "
            f"{', '.join(extra[:6])}" + (", and others" if len(extra) > 6 else ""))


def model(entry: dict, state: State, committed: dict, say) -> dict:
    """The public half: files, required paths, official voices and the config.

    No credential is taken and none is passed on. The public repository is
    public, and a run that quietly used a token to read it would be a run whose
    success said nothing about whether an ordinary installation can.
    """
    found = dict(entry)
    repo = str(entry.get("public_repo") or "")
    if not repo:
        raise PinError(f"{MODEL_ID} names no public_repo")
    commit, served = repository(repo, str(entry.get("revision") or "main"), "", say)

    files = []
    config = dict(entry.get("config") or {})
    for declared in MODEL_FILES:
        _must_serve(repo, commit, served, declared.path)
        artifact = _hub_artifact(repo, commit, declared.path, declared.local_name, "")
        artifact["about"] = declared.about
        _agree(state, committed, artifact, MODEL_ID)
        say(f"  {declared.path}: {_describe(artifact)}")
        files.append(artifact)
        if declared.config_key:
            config[declared.config_key] = declared.local_name

    voices = []
    for voice in VOICES:
        _must_serve(repo, commit, served, voice.path)
        artifact = _hub_artifact(repo, commit, voice.path,
                                 f"{voice.identifier}{STATE_SUFFIX}", "")
        _agree(state, committed, artifact, f"{MODEL_ID}/voices")
        say(f"  {voice.path}: {_describe(artifact)}")
        voices.append({
            "id": voice.identifier,
            "display_name": voice.display_name,
            "language": voice.language,
            "accent": voice.accent,
            "license": VOICE_LICENSE,
            "attribution": VOICE_ATTRIBUTION.format(display_name=voice.display_name,
                                                    repo=repo),
            "source": BLOB.format(repo=repo, revision=commit, path=voice.path),
            "artifact": artifact,
        })
    _extras(repo, served, ({voice.path for voice in VOICES}
                           | {declared.path for declared in MODEL_FILES}), say)

    found["public_commit"] = commit
    found["files"] = files
    found["required_paths"] = [declared.local_name for declared in MODEL_FILES]
    found["voices"] = voices
    found["config"] = config
    sizes = [int(one["bytes"] or 0) for one in files]
    if all(sizes):
        # Only when every file answered. A partial total shown as "about" would
        # be a download estimate that is wrong in the one direction that matters.
        found["about_bytes"] = sum(sizes) + sum(int(one["artifact"]["bytes"] or 0)
                                                for one in voices)
    state.files = len(files)
    state.voices = len(voices)
    return found


def cloning(entry: dict, state: State, committed: dict, token: str, say) -> dict:
    """The gated half: the cloning-capable weights, if the gate opens.

    Raises :class:`Gate` when it does not, which the caller turns into a
    sentence. Nothing already resolved is undone by that: the public half is a
    separate list, a separate install and a separate marker file, and section
    23.3's whole point is that Pocket can speak long before it can clone.
    """
    found = dict(entry)
    repo = str(entry.get("cloning_repo") or "")
    if not repo:
        raise PinError(f"{MODEL_ID} names no cloning_repo")
    commit, served = repository(repo, str(entry.get("cloning_revision")
                                           or entry.get("revision") or "main"), token, say)

    files = []
    config = dict(found.get("config") or {})
    for declared in CLONING_FILES:
        _must_serve(repo, commit, served, declared.path)
        artifact = _hub_artifact(repo, commit, declared.path, declared.local_name, token)
        artifact["about"] = declared.about
        # Written as well as forced by the reader. ``mc_voice_pocket.bundle``
        # marks every cloning file authorized by construction, and the flag is
        # here too so that the manifest says on its face which artifacts need a
        # credential rather than leaving it to be inferred from a list's name.
        artifact["authorized"] = True
        _agree(state, committed, artifact, f"{MODEL_ID}/cloning")
        say(f"  {declared.path}: {_describe(artifact)}")
        files.append(artifact)
        if declared.config_key:
            config[declared.config_key] = declared.local_name

    found["cloning_commit"] = commit
    found["cloning_files"] = files
    found["cloning_required_paths"] = [declared.local_name for declared in CLONING_FILES]
    found["config"] = config
    state.cloning = len(files)
    return found


# --------------------------------------------------------------------------- #
# The whole manifest
# --------------------------------------------------------------------------- #


def build(existing: dict, want_model: bool, token: str, say) -> tuple:
    """``(manifest, state)`` -- the closure refreshed and everything else kept.

    A deep copy is edited rather than the document that was read, so a run that
    ends in a refusal has not half-modified the thing it refused to write.
    """
    state = State()
    committed = _committed(existing)
    found = json.loads(json.dumps(existing)) if existing else {}
    if int(found.get("schema") or 0) not in (0, SCHEMA):
        raise PinError(f"the manifest is schema {found.get('schema')} and this tool writes "
                       f"schema {SCHEMA}")
    found["schema"] = SCHEMA

    entries = closure(found)
    runtime = dict(found.get("runtime") or {})
    _self_consistent(runtime, entries)
    say(f"\nResolving {len(entries)} package(s) for Windows x86-64, Python "
        f"{', '.join(PYTHONS)}:")
    runtime["platforms"] = platforms(entries, list(runtime.get("platforms") or ()),
                                     state, committed, say)
    found["runtime"] = runtime
    state.pinned = _complete(runtime["platforms"])

    if want_model:
        models = dict(found.get("models") or {})
        entry = models.get(MODEL_ID)
        if not isinstance(entry, dict):
            raise PinError(f"the manifest has no model called {MODEL_ID!r}")
        say(f"\nResolving {MODEL_ID} from the hub:")
        # Both halves are caught, and neither discards what the other resolved.
        # An access answer is a fact about a licence rather than about the run,
        # so a gate never costs a maintainer the closure they just read from
        # PyPI -- it costs them the list the gate was in front of, and says so.
        try:
            entry = model(entry, state, committed, say)
        except Gate as gate:
            state.gate = str(gate)
            say(f"\n{gate}")
            say("The public model half is unresolved and is left exactly as it was. A gate "
                "on the public repository is not a state this feature has a remedy for: "
                "check the repository's page before running this again.")
        else:
            try:
                entry = cloning(entry, state, committed, token, say)
            except Gate as gate:
                state.gate = str(gate)
                say(f"\n{gate}")
                say("The public half is resolved and will be written. PocketTTS speaks "
                    "with its official voices; cloning stays a separate state until the "
                    "gate opens.")
        models[MODEL_ID] = entry
        found["models"] = models

    found["pinned"] = state.pinned
    # The note describes a state, so it is rewritten when the state changes. A
    # manifest that still said PROVISIONAL over a resolved closure would be a
    # trust root lying about itself, which is the one thing it cannot do.
    found["notes"] = NOTE_PINNED if state.pinned else NOTE_PROVISIONAL
    found["version"] = int(found.get("version") or 0) + 1
    return found, state


def survey(existing: dict, path: Path, say) -> None:
    """What the manifest says about itself, before anything is asked of anybody.

    Printed first and without a network call, because the commonest reason to
    run this tool is to find out what state a checkout is in -- and a machine
    behind a proxy should learn that much before it learns it cannot reach
    pypi.org.
    """
    runtime = existing.get("runtime") or {}
    say(f"{path} as it stands:")
    say(f"  pinned: {'yes' if existing.get('pinned') else 'no'}")
    say(f"  runtime.closure: {len(runtime.get('closure') or ())} package(s) written down")
    for item in runtime.get("platforms") or ():
        artifacts = item.get("artifacts") or ()
        hashed = sum(1 for one in artifacts if one.get("sha256"))
        say(f"  {item.get('id')}: "
            + (f"{len(artifacts)} wheel(s), {hashed} hashed" if artifacts
               else "no wheels resolved"))
    for name, entry in (existing.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        say(f"  {name}: {len(entry.get('files') or ())} model file(s), "
            f"{len(entry.get('voices') or ())} official voice(s), "
            f"{len(entry.get('cloning_files') or ())} cloning file(s)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report what would be written and change nothing")
    parser.add_argument("--model", action="store_true",
                        help="also resolve the model, its official voices and the gated "
                             "voice-cloning weights")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    arguments = parser.parse_args(argv)

    def say(text):
        print(text, flush=True)

    if arguments.manifest.name in SIBLINGS:
        print(f"{arguments.manifest.name} belongs to another engine. This tool writes "
              f"PocketTTS's manifest and nothing else.", file=sys.stderr)
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

    token = _credential()
    if arguments.model and not token:
        say("\nNo Hugging Face token is in this environment, so the gated voice-cloning "
            "repository will refuse. The public model and its official voices do not "
            "need one.")

    try:
        found, state = build(existing, arguments.model, token, say)
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

    say(f"\n{state.wheels} wheel(s), {state.files} model file(s), {state.voices} official "
        f"voice(s) and {state.cloning} cloning file(s) resolved.")

    if arguments.check:
        say("Nothing was written.")
        return 0 if state.pinned and not state.gate else 1

    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    say(f"Wrote {arguments.manifest} (manifest version {found['version']}, "
        f"pinned {'true' if state.pinned else 'false'}).")
    if state.gate:
        say("The voice-cloning weights are still unresolved. Run this again once the "
            "conditions are accepted and a token is in the environment.")
    return 1 if state.gate or not state.pinned else 0


if __name__ == "__main__":
    raise SystemExit(main())
