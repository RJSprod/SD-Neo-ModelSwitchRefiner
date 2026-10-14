"""Resolve and pin Breeze TTS 2's closure, its inference code and its model.

Not part of the extension and never imported by it. A maintainer's tool, run on
a machine that can reach the publishers, beside ``tools/pin_vibevoice_models.py``
whose solver it uses: the closure is *solved* from what every publisher declares
about every other package, for CPython 3.13 on Windows x86-64, and written down
in full where a reviewer can read it.

    python tools/pin_breeze_models.py --check            # report, change nothing
    python tools/pin_breeze_models.py --source CHECKOUT  # solve and pin the wheels from
                                                         #   pypi.org, and Breeze's code from
                                                         #   raw.githubusercontent.com, listed
                                                         #   from a clone at the pinned commit
    python tools/pin_breeze_models.py --keep-closure     # re-pin the triples already written
    python tools/pin_breeze_models.py --model            # also list, size and hash the model
                                                         #   from huggingface.co
    python tools/pin_breeze_models.py --torch            # also record torch's and torchaudio's
                                                         #   digests from download.pytorch.org
    python tools/pin_breeze_models.py --flash-attn       # also record the community flash-attn
                                                         #   wheel's digest from huggingface.co

Exit codes as the other pinners: 0 when the manifest is complete, 1 when
something is still unresolved, 2 when a maintainer has to look first.

What is pinned, and from where
------------------------------
The PyPI closure: every wheel named, sized and hashed from pypi.org, from the
roots in :data:`ROOTS` -- ``qwen-tts`` (Breeze's audio codec) and the
``transformers`` and ``accelerate`` it pins exactly, with ``huggingface-hub``,
``numpy`` and ``soundfile`` at Breeze's own floors.

Breeze's own code: not a wheel, and a GitHub archive's bytes are generated on
request and have changed under the same commit before. So the repository's
files at :data:`COMMIT` are pinned one by one, each from
raw.githubusercontent.com -- the blob's own bytes -- with its size and SHA-256:
every file of ``breeze_infer`` the worker imports, the whole ``models``
package, ``configs/fast.json`` and the licence. ``--source`` names a local
clone at that commit to list them from (the GitHub API is not needed); every
file is then fetched from the raw host and hashed, and a clone whose bytes
disagree with the host's is refused.

Torch and torchaudio: the CUDA 12.8 builds live only on download.pytorch.org.
Each is a *versioned resolve* the installer answers against the publisher's
index on the user's machine, exactly as VibeVoice's torch is; ``--torch``
records both digests here. torchaudio is not optional: ``qwen_tts`` imports it
at module level, and a torchaudio built against another torch does not load.

Flash attention 2: the community wheel for CPython 3.13, torch 2.9.1 and CUDA
12.8 (Wildminder's AI-windows-whl, built from flash-attn 2.8.3 -- the version
Breeze's own Docker image builds) is a *declared* download, by URL. It is
checked at install time against the digest its host states and recorded in the
local overlay; ``--flash-attn`` turns that into a committed constant.

The model: ``BreezeBlue/Breeze-TTS-2``. Declared by repository and revision;
the installer lists the files at install time from the hub's own tree and
checks each against the digest the hub states. ``--model`` lists them here at a
named commit with their sizes and SHA-256s (git blob ids for the small files
that are not LFS objects).

It will not change a digest that is already checked in, and it will not open
another engine's manifest (:data:`SIBLINGS`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pin_vibevoice_models as base  # noqa: E402  (a sibling tool, by path)

PinError = base.PinError
PyPI = base.PyPI

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "voice" / "managed-breeze-models.json"

SIBLINGS = base.SIBLINGS + ("managed-vibevoice-models.json",)
"""The manifests this tool refuses to open, by name."""

SCHEMA = 1
USER_AGENT = "ModelChain-Breeze-Pinner"
PYTHON = base.PYTHON
PLATFORM_ID = "windows-x86_64-cp313-cu128"
MODEL_ID = "breeze-tts-2"

REPO = "breezeblue-ai/breeze-tts"
COMMIT = "6eba901d1845353802652c708a4740fdfc5baa75"
"""Breeze's inference code, at the commit this build was written and reviewed
against (2026-10-13). A newer commit is a review decision: its templates and
streaming runtime are what the worker binds to."""

RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"

SOURCE_FILES = ("breeze_infer/__init__.py", "breeze_infer/audio.py",
                "breeze_infer/runtime.py", "breeze_infer/templates.py",
                "configs/fast.json", "LICENSE")
SOURCE_TREES = ("models/",)
"""What of the repository the runtime carries: the four ``breeze_infer`` modules
the worker imports (not its HTTP server or batch endpoint), the whole ``models``
package, the fast path's profile and the licence. Not the tests, the Docker
files, the assets or ``infer.py``."""

ROOTS = (
    ("qwen-tts", "==0.1.1"),
    ("transformers", "==4.57.3"),
    ("accelerate", "==1.12.0"),
    ("huggingface-hub", ">=0.34"),
    ("numpy", ">=2.0"),
    ("soundfile", ">=0.13"),
)
"""Where the solve starts: ``qwen-tts`` at the release Breeze's requirements
name, ``transformers`` and ``accelerate`` at the versions ``qwen-tts`` pins in
its own metadata, and Breeze's floors for the rest. Everything else is derived."""

EXCLUDED = {
    "gradio": "declared by qwen-tts and imported only by its demo (qwen_tts/cli/demo.py), "
              "which nothing the worker imports reaches; read against the 0.1.1 wheel",
    "torchaudio": "resolved from the publisher's cu128 index at install time, beside torch, "
                  "because the PyPI build is built against another torch and qwen_tts "
                  "imports torchaudio at module level",
    "hf-xet": "huggingface-hub declares it behind a marker that spells the machine "
              "\"amd64\" in lower case while Windows reports \"AMD64\", so pip does not "
              "install it there either; it accelerates downloads the worker never makes",
}
"""Declared dependencies this closure deliberately does not ship, and why. Breeze's
own ``requirements.txt`` also names fastapi, uvicorn, python-multipart, pytest and
ruff -- its HTTP server and its tests -- which no package in the closure declares
and the worker never imports; the manifest's notes say so."""

NOT_FROM_REQUIREMENTS = {
    "fastapi": "Breeze's HTTP server (breeze_infer.api), which is not run: the worker "
               "answers over a pipe",
    "uvicorn": "the HTTP server's server",
    "python-multipart": "the HTTP server's form parsing",
    "pytest": "Breeze's tests",
    "ruff": "Breeze's linter",
}

TORCH = {"index": "https://download.pytorch.org/whl/cu128/torch/", "package": "torch",
         "version": "2.9.1", "about_bytes": 2_900_000_000}
TORCHAUDIO = {"index": "https://download.pytorch.org/whl/cu128/torchaudio/",
              "package": "torchaudio", "version": "2.9.1", "about_bytes": 2_000_000}

FLASH_ATTN = {
    "filename": "flash_attn-2.8.3+cu128torch2.9.1cxx11abiTRUE-cp313-cp313-win_amd64.whl",
    "url": ("https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/"
            "flash_attn-2.8.3%2Bcu128torch2.9.1cxx11abiTRUE-cp313-cp313-win_amd64.whl"),
    "version": "2.8.3",
    "source": "Wildminder/AI-windows-whl (github.com/wildminder/AI-windows-whl, wheels.json)",
    "about_bytes": 260_000_000,
}
"""The community Flash attention 2 wheel for exactly this closure: flash-attn 2.8.3
built for CPython 3.13, torch 2.9.1 and CUDA 12.8 on Windows x86-64."""

HUB_TREE = "https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
HUB_REVISION = "https://huggingface.co/api/models/{repo}/revision/{revision}"
HUB_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


# --------------------------------------------------------------------------- #
# Breeze's own code
# --------------------------------------------------------------------------- #


def listed_source(checkout: Path) -> list:
    """The files the runtime carries, from a clone, checked to be at :data:`COMMIT`."""
    try:
        head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        names = subprocess.run(["git", "-C", str(checkout), "ls-tree", "-r", "--name-only",
                                COMMIT], capture_output=True, text=True,
                               check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PinError(f"{checkout} is not a clone of {REPO} holding {COMMIT[:12]} ({exc})") \
            from None
    if head != COMMIT:
        raise PinError(f"{checkout} is at {head[:12]}, not {COMMIT[:12]}; check out the "
                       f"pinned commit first")
    wanted = [name for name in names
              if name in SOURCE_FILES or any(name.startswith(tree) for tree in SOURCE_TREES)]
    missing = [name for name in SOURCE_FILES if name not in wanted]
    if missing:
        raise PinError(f"{REPO} at {COMMIT[:12]} has no {missing[0]}")
    return sorted(wanted)


def source_file(path: str, checkout: Path, say) -> dict:
    """One file as the raw host serves it at the commit, hashed, and checked against the clone."""
    url = RAW.format(repo=REPO, commit=COMMIT, path=urllib.parse.quote(path))
    status, body = base._get(url)
    if status != 200:
        raise PinError(f"{url} answered HTTP {status}")
    # An empty file is a file (``models/__init__.py`` is one): it is pinned at
    # zero bytes and the digest of nothing, and the clone has to agree.
    body = body or b""
    local = (checkout / path).read_bytes()
    if local != body:
        raise PinError(f"{path} in {checkout} is not what {REPO} serves at {COMMIT[:12]}")
    digest = hashlib.sha256(body).hexdigest()
    say(f"  {path}: {len(body)} bytes, sha256 {digest[:12]}")
    return {"path": path, "url": url, "bytes": len(body), "sha256": digest}


# --------------------------------------------------------------------------- #
# The model, from the hub's own tree
# --------------------------------------------------------------------------- #


def model(entry: dict, say) -> dict:
    """The model's files at a named commit, each with its size and digest."""
    found = dict(entry)
    repo = str(entry.get("repo") or "")
    revision = str(entry.get("revision") or "main")
    status, body = base._get(HUB_REVISION.format(repo=repo, revision=revision))
    if status != 200:
        raise PinError(f"{repo}: the hub answered {status} for revision {revision}")
    commit = str(json.loads(body.decode("utf-8", "replace")).get("sha") or "")
    if len(commit) != 40:
        raise PinError(f"{repo}: {commit!r} is not a commit sha")
    status, body = base._get(HUB_TREE.format(repo=repo, revision=commit))
    if status != 200:
        raise PinError(f"{repo}: the hub answered {status} for its file tree")
    files = []
    skip = set(entry.get("skip") or ())
    suffixes = tuple(entry.get("skip_suffixes") or ())
    for item in json.loads(body.decode("utf-8", "replace")):
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        path = str(item.get("path") or "")
        if path in skip or path.lower().endswith(suffixes):
            continue
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else None
        row = {"path": path,
               "url": HUB_RESOLVE.format(repo=repo, revision=commit,
                                         path=urllib.parse.quote(path)),
               "bytes": int((lfs or item).get("size") or 0)}
        if lfs:
            row["sha256"] = str(lfs.get("oid") or "").casefold()
        else:
            row["git_oid"] = str(item.get("oid") or "").casefold()
        say(f"  {path}: {row['bytes']} bytes")
        files.append(row)
    if not any(row["path"] == "config.json" for row in files):
        raise PinError(f"{repo} at {commit[:12]} has no config.json")
    if not any(row["path"].startswith("audio_tokenizer/") for row in files):
        raise PinError(f"{repo} at {commit[:12]} has no audio_tokenizer folder, which Breeze's "
                       f"loader refuses to start without")
    found["commit"] = commit
    found["files"] = sorted(files, key=lambda row: row["path"])
    weights = sum(row["bytes"] for row in files if row["path"].endswith(".safetensors"))
    if weights:
        estimates = dict(found.get("estimates") or {})
        estimates["weights_bytes"] = weights
        found["estimates"] = estimates
    return found


def _model_hashed(entry: dict) -> bool:
    files = list(entry.get("files") or ())
    return bool(files) and all(int(row.get("bytes") or 0) > 0
                               and (len(str(row.get("sha256") or "")) == 64
                                    or len(str(row.get("git_oid") or "")) == 40)
                               for row in files)


# --------------------------------------------------------------------------- #
# The whole manifest
# --------------------------------------------------------------------------- #


@dataclass
class State:
    disagreements: list = field(default_factory=list)
    wheels: int = 0
    closure_complete: bool = False
    source_pinned: bool = False
    resolves_recorded: bool = False
    flash_recorded: bool = False
    model_hashed: bool = False

    @property
    def pinned(self) -> bool:
        return bool(self.closure_complete and self.source_pinned and self.resolves_recorded
                    and self.flash_recorded and self.model_hashed)


def _resolve_entry(resolve: dict, previous: dict, record: bool, state: State, say) -> dict:
    found = {key: resolve[key] for key in ("index", "package", "version")}
    for key in ("filename", "sha256", "bytes"):
        if previous.get(key):
            found[key] = previous[key]
    if record:
        wheel = base.torch_wheel(found, say)
        if found.get("sha256") and found["sha256"] != wheel["sha256"]:
            state.disagreements.append(f"{found['package']}/{wheel['filename']}: checked in "
                                       f"{found['sha256']}, published {wheel['sha256']}")
        found.update({"filename": wheel["filename"], "sha256": wheel["sha256"],
                      "bytes": wheel["bytes"]})
    return found


def _flash_entry(previous: dict, record: bool, state: State, say) -> dict:
    found = {"filename": FLASH_ATTN["filename"], "local_name": FLASH_ATTN["filename"],
             "url": FLASH_ATTN["url"], "bytes": previous.get("bytes"),
             "sha256": previous.get("sha256"), "declared": True,
             "version": FLASH_ATTN["version"], "source": FLASH_ATTN["source"]}
    if record:
        status, headers = base._head(FLASH_ATTN["url"])
        if status >= 400:
            raise PinError(f"the flash-attn wheel answered HTTP {status}")
        digest = base._published_sha256(headers)
        size = base._published_size(headers)
        if not digest or not size:
            raise PinError("the host stated no digest or size for the flash-attn wheel")
        if found.get("sha256") and found["sha256"] != digest:
            state.disagreements.append(f"flash-attn: checked in {found['sha256']}, published "
                                       f"{digest}")
        found.update({"sha256": digest, "bytes": size})
        say(f"  {FLASH_ATTN['filename']}: {size} bytes, sha256 {digest[:12]}")
    return found


def notes_for(state: State) -> str:
    """The manifest's own account of what is and is not pinned. Always current."""
    parts = [f"{'PINNED' if state.pinned else 'PARTIALLY PINNED'}."]
    parts.append("The PyPI closure is " + (
        "resolved: every wheel is named, sized and hashed from pypi.org, solved from the "
        "packages' own declarations for CPython 3.13 on Windows x86-64 by "
        "tools/pin_breeze_models.py, which refuses a closure whose pins contradict one "
        "another." if state.closure_complete else "not yet resolved."))
    parts.append("Breeze's own code is " + (
        f"pinned file by file at {COMMIT} from raw.githubusercontent.com, each with its size "
        f"and SHA-256." if state.source_pinned else "not yet pinned."))
    parts.append("torch 2.9.1 and torchaudio 2.9.1 are NOT PyPI wheels here: the CUDA 12.8 "
                 "builds live only on download.pytorch.org, so each is a versioned resolve "
                 "the installer answers against the publisher's own index on the user's "
                 "machine, taking the cp313 win_amd64 wheel of exactly that version and the "
                 "SHA-256 the index states" + (
                     ", and this file records both digests." if state.resolves_recorded else
                     "; the machine that wrote this could not reach download.pytorch.org, so "
                     "`python tools/pin_breeze_models.py --torch` on one that can turns the "
                     "index's word into committed constants."))
    parts.append("Flash attention 2 is installed with the runtime, from the community wheel "
                 f"{FLASH_ATTN['filename']} ({FLASH_ATTN['source']})" + (
                     "; its digest is recorded here." if state.flash_recorded else
                     ", checked at install time against the digest its host states and "
                     "recorded in voice/managed-breeze-models.local.json; `--flash-attn` "
                     "records it here."))
    parts.append("The model is " + (
        "PINNED: every file at a named commit with its size and digest." if
        state.model_hashed else
        "DECLARED BUT NOT HASHED: its repository and revision are written down, the installer "
        "lists the files from the hub's own tree at install time and checks every one against "
        "the digest the hub states (the LFS SHA-256, or the git blob id of a small file), and "
        "records them in voice/managed-breeze-models.local.json; `--model` on a machine that "
        "can reach huggingface.co commits them."))
    parts.append("Not installed, on purpose: gradio (qwen-tts's demo) and Breeze's HTTP "
                 "server and tests (fastapi, uvicorn, python-multipart, pytest, ruff); "
                 "runtime.excluded records each reason.")
    return " ".join(parts)


def build(existing: dict, options, say) -> tuple:
    """``(manifest, state)`` -- the closure refreshed and everything else kept."""
    state = State()
    found = json.loads(json.dumps(existing)) if existing else {}
    if int(found.get("schema") or 0) not in (0, SCHEMA):
        raise PinError(f"the manifest is schema {found.get('schema')} and this tool writes "
                       f"schema {SCHEMA}")
    found["schema"] = SCHEMA
    runtime = dict(found.get("runtime") or {})
    declared = list(runtime.get("platforms") or ())
    if len(declared) != 1 or str(declared[0].get("id") or "") != PLATFORM_ID:
        raise PinError(f"the manifest must advertise exactly one platform, {PLATFORM_ID}")
    entry = dict(declared[0])
    committed = {str(item.get("filename")): str(item.get("sha256") or "").casefold()
                 for item in entry.get("artifacts") or ()
                 if isinstance(item, dict) and item.get("filename") and item.get("sha256")}
    pypi = options.pypi

    if options.keep_closure:
        say(f"\nRe-pinning the {len(runtime.get('closure') or ())} package(s) written down:")
        chosen = base.resolve_triples(runtime.get("closure") or (), pypi, say)
    else:
        say(f"\nSolving the closure from {len(ROOTS)} roots for {PLATFORM_ID}:")
        chosen = base.solve(ROOTS, pypi, EXCLUDED, say, torch=TORCH)
    torch_requires = base._requirements_of(pypi.release(TORCH["package"], TORCH["version"]))
    base.requirements_met(chosen, EXCLUDED, TORCH["version"], torch_requires, roots=ROOTS)
    audio_requires = base._requirements_of(pypi.release(TORCHAUDIO["package"],
                                                        TORCHAUDIO["version"]))
    for raw in audio_requires:
        if raw.replace(" ", "").startswith("torch") and TORCH["version"] not in raw:
            raise PinError(f"torchaudio {TORCHAUDIO['version']} needs {raw}, and torch is "
                           f"pinned at {TORCH['version']}")
    say("  every pin satisfies every other pin's declared requirement")

    rows = base.triples(chosen, roots=ROOTS)
    artifacts = []
    for key in base._ordered_keys(chosen, rows):
        artifact = base.wheel_artifact(chosen[key].file)
        was = committed.get(artifact["filename"])
        if was and was != artifact["sha256"]:
            state.disagreements.append(f"{artifact['filename']}: checked in {was}, published "
                                       f"{artifact['sha256']}")
        artifacts.append(artifact)
    state.wheels = len(artifacts)
    state.closure_complete = bool(artifacts)

    previous = {str((item.get("resolve") or {}).get("package") or ""): item["resolve"]
                for item in entry.get("artifacts") or ()
                if isinstance(item, dict) and isinstance(item.get("resolve"), dict)}
    if options.torch:
        say("\nReading torch and torchaudio from the publisher's index:")
    resolves = [_resolve_entry(TORCH, previous.get("torch") or {}, options.torch, state, say),
                _resolve_entry(TORCHAUDIO, previous.get("torchaudio") or {}, options.torch,
                               state, say)]
    state.resolves_recorded = all(len(str(one.get("sha256") or "")) == 64 for one in resolves)
    flash_before = next((item for item in entry.get("artifacts") or ()
                         if isinstance(item, dict) and item.get("declared")), {})
    if options.flash_attn:
        say("\nReading the flash-attn wheel from its host:")
    flash = _flash_entry(flash_before, options.flash_attn, state, say)
    state.flash_recorded = len(str(flash.get("sha256") or "")) == 64 \
        and int(flash.get("bytes") or 0) > 0
    entry["artifacts"] = (artifacts + [{"local_name": one["package"], "resolve": one}
                                       for one in resolves] + [flash])

    source = dict(runtime.get("source") or {})
    if options.source is not None:
        say(f"\nPinning {REPO} at {COMMIT[:12]} from raw.githubusercontent.com:")
        paths = listed_source(options.source)
        source_files = [source_file(path, options.source, say) for path in paths]
        before = {str(row.get("path")): str(row.get("sha256") or "")
                  for row in source.get("files") or ()}
        for row in source_files:
            if before.get(row["path"]) and before[row["path"]] != row["sha256"] \
                    and str(source.get("commit") or "") == COMMIT:
                state.disagreements.append(f"{row['path']}: checked in "
                                           f"{before[row['path']]}, served {row['sha256']}")
        source.update({"repo": REPO, "commit": COMMIT, "license": "Apache-2.0",
                       "files": source_files})
    state.source_pinned = str(source.get("commit") or "") == COMMIT and bool(
        source.get("files")) and all(len(str(row.get("sha256") or "")) == 64
                                     for row in source.get("files") or ())

    runtime["closure"] = rows
    runtime["excluded"] = dict(EXCLUDED)
    runtime["not_installed"] = dict(NOT_FROM_REQUIREMENTS)
    runtime["torch"] = dict(TORCH, **{key: resolves[0][key] for key in ("filename", "sha256",
                                                                         "bytes")
                                      if resolves[0].get(key)})
    runtime["torchaudio"] = dict(TORCHAUDIO, **{key: resolves[1][key]
                                                for key in ("filename", "sha256", "bytes")
                                                if resolves[1].get(key)})
    runtime["flash_attn"] = {key: FLASH_ATTN[key] for key in ("version", "source",
                                                              "about_bytes")}
    runtime["source"] = source
    runtime["platforms"] = [entry]
    found["runtime"] = runtime

    models = dict(found.get("models") or {})
    model_entry = models.get(MODEL_ID)
    if not isinstance(model_entry, dict):
        raise PinError(f"the manifest has no model called {MODEL_ID!r}")
    if options.model:
        say(f"\nListing {MODEL_ID} from the hub:")
        model_entry = model(model_entry, say)
        models[MODEL_ID] = model_entry
        found["models"] = models
    state.model_hashed = _model_hashed(model_entry)

    found["pinned"] = state.pinned
    found["notes"] = notes_for(state)
    found["version"] = int(found.get("version") or 0) + 1
    return found, state


class Options:
    def __init__(self, keep_closure=False, model=False, torch=False, flash_attn=False,
                 source=None, pypi=None):
        self.keep_closure = keep_closure
        self.model = model
        self.torch = torch
        self.flash_attn = flash_attn
        self.source = source
        self.pypi = pypi or PyPI()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report what would be written and change nothing")
    parser.add_argument("--keep-closure", action="store_true",
                        help="re-pin the triples already written instead of solving")
    parser.add_argument("--source", type=Path, default=None,
                        help="a clone of breezeblue-ai/breeze-tts at the pinned commit")
    parser.add_argument("--model", action="store_true",
                        help="also list, size and hash the model from the hub")
    parser.add_argument("--torch", action="store_true",
                        help="also record torch's and torchaudio's digests from the cu128 index")
    parser.add_argument("--flash-attn", action="store_true",
                        help="also record the community flash-attn wheel's digest")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    arguments = parser.parse_args(argv)

    def say(text):
        print(text, flush=True)

    if arguments.manifest.name in SIBLINGS:
        print(f"{arguments.manifest.name} belongs to another engine. This tool writes "
              f"Breeze's manifest and nothing else.", file=sys.stderr)
        return 2
    try:
        existing = (json.loads(arguments.manifest.read_text(encoding="utf-8"))
                    if arguments.manifest.exists() else {})
    except (OSError, ValueError) as exc:
        print(f"{arguments.manifest} could not be read ({exc})", file=sys.stderr)
        return 2
    try:
        found, state = build(existing, Options(keep_closure=arguments.keep_closure,
                                               model=arguments.model, torch=arguments.torch,
                                               flash_attn=arguments.flash_attn,
                                               source=arguments.source), say)
    except PinError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    if state.disagreements:
        print("\nThe published artifacts no longer match what is checked in:",
              file=sys.stderr)
        for line in state.disagreements:
            print(f"  {line}", file=sys.stderr)
        print("\nNothing was written.", file=sys.stderr)
        return 2
    say(f"\n{state.wheels} wheel(s) resolved; source "
        f"{'pinned' if state.source_pinned else 'not pinned'}; torch/torchaudio digests "
        f"{'recorded' if state.resolves_recorded else 'not recorded'}; flash-attn digest "
        f"{'recorded' if state.flash_recorded else 'not recorded'}; model "
        f"{'hashed' if state.model_hashed else 'declared but not hashed'}.")
    body = json.dumps(found, indent=2, ensure_ascii=False) + "\n"
    if arguments.check:
        current = (arguments.manifest.read_text(encoding="utf-8")
                   if arguments.manifest.exists() else "")
        same = base._without_version(current) == base._without_version(body)
        say("Nothing was written." + ("" if same else " The manifest would change."))
        return 0 if same and state.pinned else 1
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(body, encoding="utf-8")
    say(f"Wrote {arguments.manifest} (manifest version {found['version']}, pinned "
        f"{'true' if state.pinned else 'false'}).")
    return 0 if state.pinned else 1


if __name__ == "__main__":
    raise SystemExit(main())
