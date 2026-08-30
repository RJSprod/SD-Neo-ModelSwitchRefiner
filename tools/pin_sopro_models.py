"""Resolve and pin the Sopro V2 Windows CPU closure from PyPI.

Not part of the extension and never imported by it. A maintainer's tool, run on
a machine that can reach pypi.org, which is why it lives in ``tools/`` rather
than in ``scripts/`` where Forge would import it at start-up. It is the sibling
of ``tools/pin_voice_models.py`` and it does the job that one deliberately does
not: it writes the *runtime* closure.

Why a tool rather than a hand-written manifest
----------------------------------------------
Sopro's own metadata declares dependency floors -- ``torch>=2.3``,
``torchaudio>=2.3``, ``numpy>=1.24`` -- and a floor is not a runtime identity.
Two machines that both satisfy those floors can be running different attention
kernels, different thread pools and different serialization behaviour, which is
the difference between a saved voice that reconstructs and one that does not.
So this repository pins *exact builds* and the fingerprint of every byte, and
this tool is how that list is produced without anybody transcribing eighteen
SHA-256 digests by hand.

    python tools/pin_sopro_models.py --check      # report, change nothing
    python tools/pin_sopro_models.py              # rewrite the closure

What it will not do
-------------------
It will not resolve a dependency graph. :data:`CLOSURE` is written down, in
full, by a person who read Sopro's imports -- because "whatever pip decides
today" is the property this whole design exists to remove. The tool's job is to
turn that written-down list into sizes and hashes, and to fail loudly if PyPI
no longer serves a wheel the list names.

It will not add a Linux platform. PyPI's Linux ``torch`` wheels pull the whole
CUDA closure through their dependency markers, and invariant I-9 says the first
Sopro release neither claims nor consumes a graphics device. A Linux CPU
closure comes from PyTorch's own ``+cpu`` index and is a separate, deliberate
addition with its own gate -- not something a script guesses at because it
happened to be run on Linux.

It will not touch ``voice/managed-voice-models.json``. Kokoro's closure is a
different trust root with a different lifecycle, and a tool that could edit both
is a tool that can break the working engine while installing the optional one.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "voice" / "managed-sopro-models.json"

TIMEOUT = 120.0
USER_AGENT = "ModelChain-Sopro-Pinner"

PYTHONS = ("3.10", "3.11", "3.12", "3.13")
"""The Python minors this release advertises for Sopro on Windows.

An allowlist rather than a range, and section 17's point exactly: PyTorch's
general Windows support establishes feasibility, and a *combination* becomes
supported only after the exact closure below has been installed and self-tested
on that class. Adding 3.14 here is a release decision, not a version bump.
"""

PURE = "pure"
BINARY = "binary"
ABI3 = "abi3"
"""How to recognise the one wheel that belongs to a platform.

``PURE``  -- ``py3-none-any`` (or ``py2.py3-none-any``), one file for everything.
``BINARY``-- ``cp310-cp310-win_amd64``, one file per Python minor.
``ABI3``  -- ``cp310-abi3-win_amd64``, built once against the stable ABI and
             valid for every later minor, which is what safetensors ships.
"""

CLOSURE = (
    # The engine itself.
    ("sopro", "2.0.5", PURE),
    # What Sopro imports directly: model.py, audio.py, hub.py and text.py.
    ("torch", "2.11.0", BINARY),
    ("torchaudio", "2.11.0", BINARY),
    ("numpy", "2.2.6", BINARY),
    ("safetensors", "0.8.0", ABI3),
    ("sentencepiece", "0.2.2", BINARY),
    ("soundfile", "0.14.0", "windows"),
    ("cffi", "2.0.0", BINARY),
    ("pycparser", "2.23", PURE),
    # What Torch itself declares on Windows. Its CUDA dependencies are all
    # marked ``platform_system == "Linux"`` and are correctly absent here --
    # which is why this closure is 130 MB rather than three gigabytes, and why
    # a Windows install cannot acquire a graphics runtime by accident.
    ("filelock", "3.20.0", PURE),
    ("typing_extensions", "4.15.0", PURE),
    ("setuptools", "81.0.0", PURE),
    ("sympy", "1.14.0", PURE),
    ("mpmath", "1.3.0", PURE),
    ("networkx", "3.4.2", PURE),
    ("jinja2", "3.1.6", PURE),
    ("MarkupSafe", "3.0.3", BINARY),
    ("fsspec", "2025.9.0", PURE),
)
"""Every wheel the Sopro worker's interpreter gets, and nothing else.

Written out rather than resolved, and deliberately shorter than a `pip install
sopro` would produce. ``huggingface-hub`` is absent because the production load
path passes Sopro a verified local directory and ``resolve_artifacts`` only
imports the hub when it is handed a repo id -- so leaving it out is not a
saving, it is section 54 enforced by absence: a worker with no hub client
cannot resolve a model from the Internet even if something asked it to.
"""

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "semantic_encoder.safetensors",
    "speaker_encoder.safetensors",
    "vocoder.safetensors",
    "vocoder_streaming.safetensors",
    "tokenizer.model",
)
"""``sopro.hub.ARTIFACT_FILES``, repeated here on purpose.

The manifest has to name what an installed model bundle must contain without
importing Sopro to ask, because the machine writing the manifest is not the
machine that has the runtime. ``tests/test_voice_sopro.py`` asserts this tuple
matches what the pinned Sopro source declares.
"""

MODEL_REPO = "samuel-vitorino/sopro-v2-turbo"
MODEL_REVISION = "main"


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
        elif kind == "windows":
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


def platforms(say) -> list:
    """One fully pinned wheel closure per advertised Python minor."""
    releases = {}
    for name, version, _kind in CLOSURE:
        say(f"reading {name} {version} from PyPI")
        releases[(name, version)] = _release(name, version)

    found = []
    for python in PYTHONS:
        artifacts = []
        for name, version, kind in CLOSURE:
            artifacts.append(_artifact(_wheel(releases[(name, version)], kind, python)))
        total = sum(item["bytes"] for item in artifacts)
        say(f"windows-x86_64-cp{python.replace('.', '')}: {len(artifacts)} wheels, "
            f"{total / 1e6:.0f} MB")
        found.append({
            "id": f"windows-x86_64-cp{python.replace('.', '')}",
            "system": "windows",
            "machines": ["amd64", "x86_64"],
            "python": python,
            "artifacts": artifacts,
        })
    return found


def build(existing: dict, say) -> dict:
    """The whole manifest, with the closure refreshed and everything else kept."""
    found = dict(existing) if existing else {}
    found["schema"] = 1
    found["version"] = int(found.get("version") or 0) + 1
    runtime = dict(found.get("runtime") or {})
    runtime.update({
        "package": "sopro",
        "import_name": "sopro",
        "version": next(v for n, v, _ in CLOSURE if n == "sopro"),
        "torch_version": next(v for n, v, _ in CLOSURE if n == "torch"),
        "torchaudio_version": next(v for n, v, _ in CLOSURE if n == "torchaudio"),
        "provider": "cpu",
        "license": ("Apache-2.0 (Sopro, Samuel Vitorino); BSD-3-Clause (PyTorch, torchaudio, "
                    "NumPy); Apache-2.0 (safetensors, sentencepiece); BSD-3-Clause "
                    "(soundfile, cffi, pycparser, MarkupSafe, networkx); BSD-3-Clause "
                    "(Jinja2, fsspec); Unlicense (filelock); MIT (setuptools, sympy, mpmath, "
                    "typing-extensions)"),
        "platforms": platforms(say),
    })
    found["runtime"] = runtime
    found.setdefault("defaults", {"tts": "sopro-v2-turbo-cpu"})
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report what would be written and change nothing")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    arguments = parser.parse_args(argv)

    def say(text):
        print(text, flush=True)

    try:
        existing = (json.loads(arguments.manifest.read_text(encoding="utf-8"))
                    if arguments.manifest.exists() else {})
    except (OSError, ValueError) as exc:
        print(f"{arguments.manifest} could not be read ({exc})", file=sys.stderr)
        return 2

    try:
        found = build(existing, say)
    except PinError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    if arguments.check:
        print("\nNothing was written.")
        return 0

    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {arguments.manifest} (manifest version {found['version']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
