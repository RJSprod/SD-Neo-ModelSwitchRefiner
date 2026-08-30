"""Resolve and pin the DeepFilterNet Windows CPU closure from PyPI.

A maintainer's tool, never imported by the extension, and the sibling of
``tools/pin_sopro_models.py``. Run it on a machine that can reach pypi.org:

    python tools/pin_cleanup_models.py --check      # report, change nothing
    python tools/pin_cleanup_models.py              # rewrite the closure

Why this engine has an interpreter of its own
---------------------------------------------
DeepFilterNet's inference path imports Torch, and its DSP -- the ERB filterbank
and the deep-filtering stage -- lives in a Rust extension, ``DeepFilterLib``,
whose newest published wheels are for CPython 3.11. Forge runs on whatever
Python the user installed it under, which is 3.13 on the machine this was
written for, so the cleanup engine cannot share either the host's interpreter or
Sopro's cp313 Torch. It gets a cp311 interpreter and a cp311 Torch of its own.

That is the whole cost of this feature and it is worth stating plainly: about
150 MB, most of it a second Torch, to clean a twenty-second clip. The
alternative was a denoiser that gates on a speech classifier, and a reference
recording is arbitrary material somebody brings from outside.

Which Torch, and why not the newest
-----------------------------------
``df/io.py`` imports ``torchaudio.backend.common.AudioMetaData``, which
torchaudio removed after 2.2. So this pins the *newest pair that still has the
API DeepFilterNet was written against* rather than the newest pair that exists,
and a bump is a decision somebody makes after reading the diff rather than one
this tool takes on a Tuesday.

What it will not do
-------------------
It will not resolve a dependency graph: :data:`CLOSURE` is written down by a
person who read the imports. It will not add a Linux platform, for the reason
the Sopro pinner gives -- PyPI's Linux Torch pulls the entire CUDA closure. And
it will not pin the interpreter or the model, because neither is served from
PyPI: both are named here with a URL and no hash, resolved against the publisher
over HTTPS at install time and recorded, which is what ``mc_voice_models``
already does for every artifact nobody has pinned yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "voice" / "managed-cleanup-models.json"

TIMEOUT = 120.0
USER_AGENT = "ModelChain-Cleanup-Pinner"

PYTHON = "3.11"
"""The one Python minor this engine runs on.

Not a range and not the host's. ``DeepFilterLib`` publishes cp38 through cp311
and nothing newer, and 3.11 is the newest of those -- so this is the single
combination the engine is built and tested for, and an interpreter is shipped to
guarantee it rather than hoped for.
"""

PURE = "pure"
BINARY = "binary"
WINDOWS = "windows"
CPNONE = "cp-none"
"""A wheel built for one CPython but with a stable ABI tag of ``none``.

DeepFilterLib's Windows wheels are tagged ``cp311-none-win_amd64``: built
against 3.11 but declaring no ABI, because the Rust extension talks to Python
through the limited API on that platform. Named as its own kind rather than
guessed at, so a publisher who changes the tag breaks the pin loudly.
"""

CLOSURE = (
    # The engine, and the Rust extension that is the reason for all of this.
    ("deepfilternet", "0.5.6", PURE),
    ("deepfilterlib", "0.5.6", CPNONE),
    # What DeepFilterNet imports directly: enhance.py, io.py, model.py, utils.py.
    ("torch", "2.2.2", BINARY),
    ("torchaudio", "2.2.2", BINARY),
    ("numpy", "1.26.4", BINARY),
    ("loguru", "0.7.3", PURE),
    ("appdirs", "1.4.4", PURE),
    ("requests", "2.32.3", PURE),
    # loguru's own, on Windows only, and both are why this closure is not the
    # same list on another platform.
    ("colorama", "0.4.6", PURE),
    ("win32_setctime", "1.2.0", PURE),
    # requests'.
    ("charset_normalizer", "3.4.0", BINARY),
    ("idna", "3.10", PURE),
    ("urllib3", "2.2.3", PURE),
    ("certifi", "2024.8.30", PURE),
    # Torch's, on Windows. Its CUDA dependencies are all marked
    # ``platform_system == "Linux"`` and are correctly absent, which is why this
    # is 150 MB rather than three gigabytes.
    ("filelock", "3.16.1", PURE),
    ("typing_extensions", "4.12.2", PURE),
    ("sympy", "1.13.3", PURE),
    ("mpmath", "1.3.0", PURE),
    ("networkx", "3.4.2", PURE),
    ("jinja2", "3.1.4", PURE),
    ("MarkupSafe", "2.1.5", BINARY),
    ("fsspec", "2024.9.0", PURE),
    ("setuptools", "75.1.0", PURE),
)
"""Every wheel the cleanup worker's interpreter gets, and nothing else.

Shorter than ``pip install deepfilternet`` would produce. ``icecream`` and
``ptflops`` are absent because every import of either is inside a function on a
debug or training path; ``soundfile`` is absent because the worker hands
DeepFilterNet tensors rather than files and never opens one.
"""

INTERPRETER = {
    "filename": "python-3.11.9-embed-amd64.zip",
    "url": "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",
    "about_bytes": 10 * 1024 * 1024,
}
"""The interpreter the closure above is built for.

The embeddable distribution rather than an installer: it is a zip, it needs no
administrator and writes nothing outside the folder it is unpacked into, and it
has no pip -- which suits an installer that has never used one. What it also has
no ``venv``, so the layout is written directly and ``python311._pth`` is
rewritten to put the closure's site-packages on the path.

Unpinned, deliberately and with the owner's agreement: python.org is not
reachable from the workspace this manifest is generated in, so the digest is
recorded on the machine that first fetches it rather than checked against a
constant somebody reviewed. That is weaker than every wheel above and is the one
place in this feature where it is true.
"""

MODEL = {
    "identifier": "deepfilternet3",
    "label": "DeepFilterNet3 (CPU)",
    "filename": "DeepFilterNet3.zip",
    "url": "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip",
    "about_bytes": 3 * 1024 * 1024,
    "license": "MIT / Apache-2.0 (DeepFilterNet)",
    "attribution": "DeepFilterNet by Hendrik Schröter et al.",
    "expects": ["config.ini", "checkpoints"],
}
"""The trained model, as the one archive its author publishes.

Installed into a directory this extension owns and handed to ``init_df`` as a
path, so nothing resolves anything after installation -- the same rule Sopro's
model follows, and the reason neither worker needs a network client.
"""


class PinError(RuntimeError):
    """Something a maintainer has to look at rather than re-run."""


def _release(name: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            return json.load(answer)
    except Exception as exc:
        raise PinError(f"{name} {version} could not be read from PyPI ({exc})") from None


def _wheel(entry: dict, kind: str) -> dict:
    """The one wheel from a release that belongs to cp311 on Windows x86-64.

    Refuses ambiguity rather than picking, for the reason the Sopro pinner
    gives: quietly taking the first candidate is how a free-threaded build ends
    up pinned as the ordinary one.
    """
    tag = "cp" + PYTHON.replace(".", "")
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
        elif kind == CPNONE:
            if name.endswith(f"-{tag}-none-win_amd64.whl"):
                found.append(item)
        elif name.endswith(f"-{tag}-{tag}-win_amd64.whl"):
            found.append(item)
    if not found:
        raise PinError(f"{entry['info']['name']} {entry['info']['version']} has no {kind} "
                       f"wheel for Python {PYTHON} on Windows x86-64")
    if len(found) > 1:
        names = ", ".join(sorted(str(item.get("filename")) for item in found))
        raise PinError(f"{entry['info']['name']} {entry['info']['version']} offers more than "
                       f"one candidate: {names}")
    return found[0]


def _artifact(item: dict) -> dict:
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
        "sha256": sha256,
    }


def closure(say) -> list:
    found = []
    for name, version, kind in CLOSURE:
        say(f"  {name} {version}")
        found.append(_artifact(_wheel(_release(name, version), kind)))
    return found


def build(say) -> dict:
    say("Resolving the DeepFilterNet closure for Windows x86-64, Python " + PYTHON)
    wheels = closure(say)
    total = sum(item["bytes"] for item in wheels)
    return {
        "schema": 1,
        "version": 1,
        "notes": ("The cleanup engine: DeepFilterNet on its own cp311 interpreter, because "
                  "DeepFilterLib publishes no wheel for the Python Forge runs on. Installed "
                  "and removed on its own, started only while a recording is being cleaned, "
                  "and ended with the WebUI."),
        "runtime": {
            "python": PYTHON,
            "platform": "windows-x86_64-cp311",
            "build": 1,
            "interpreter": dict(INTERPRETER),
            "wheels": wheels,
            "bytes": total,
        },
        "model": dict(MODEL),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report what would change and write nothing")
    found = parser.parse_args(argv)
    say = (lambda text: print(text, file=sys.stderr))
    try:
        built = build(say)
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(built, indent=2) + "\n"
    existing = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
    if text == existing:
        say("The closure is already pinned exactly as PyPI serves it.")
        return 0
    if found.check:
        say(f"{MANIFEST.name} would change ({len(built['runtime']['wheels'])} wheels, "
            f"{built['runtime']['bytes'] / 1e6:.1f} MB).")
        return 1
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(text, encoding="utf-8")
    say(f"Wrote {MANIFEST.name}: {len(built['runtime']['wheels'])} wheels, "
        f"{built['runtime']['bytes'] / 1e6:.1f} MB, plus an interpreter and a model "
        f"resolved at install time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
