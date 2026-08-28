"""Pin the byte count and SHA-256 of every Voice Chat model artifact.

Not part of the extension and never imported by it. This is a maintainer's tool,
run on a machine that can reach huggingface.co and github.com -- which is why it
lives in ``tools/`` rather than in ``scripts/`` where Forge would import it at
startup.

What it is for
--------------
``voice/managed-voice-models.json`` ships with the runtime closure fully pinned
-- sixteen wheels, real sizes, real hashes, read from PyPI -- and with the two
*model* bundles carrying ``"bytes": null`` and ``"sha256": null``. That is not
a shortcut, it is the safe state: :mod:`mc_voice_models` refuses to download an
unpinned artifact at all, so an unpinned build is one where Voice Chat offers
the manual install instead of fetching something nobody checked.

This turns that into a pinned build. It downloads each declared artifact once,
hashes it, and writes the size and the hash into
``voice/managed-voice-models.local.json`` -- an untracked overlay beside the
manifest, so the pins survive a ``git pull`` and never turn into a merge
conflict in a file full of hashes. That is Gate 0 of the design intent: the
artifacts are not pinned until somebody has actually fetched them and looked.

    python tools/pin_voice_models.py             # write the pins
    python tools/pin_voice_models.py --check     # report, change nothing

What it will not do
-------------------
It never *changes* a hash that is already there. The hash in the repository is
the trust root for the whole feature, so a disagreement between what is checked
in and what the publisher is serving today fails the whole run and prints both.
A publisher whose files have really changed is a review decision -- somebody has
to look at the new model card and decide -- and not something a script gets to
make at three in the morning.

It also does not pin the runtime wheels. Those are already pinned, and re-reading
them from PyPI would be this tool inventing a reason to touch bytes that are
already the trust root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "voice" / "managed-voice-models.json"
OVERLAY = MANIFEST.with_name("managed-voice-models.local.json")

TIMEOUT = 120.0
CHUNK = 1024 * 256
USER_AGENT = "ModelChain-VoiceChat-Pinner"


class PinError(RuntimeError):
    """Something a maintainer has to look at rather than re-run."""


def measure(url: str, say) -> tuple[int, str]:
    """Download ``url`` once and report ``(bytes, sha256)``.

    Streamed and never kept: this is several hundred megabytes of ONNX and the
    only thing wanted from it is a hash. Writing it to disk would leave a copy
    of a model in a checkout, which is exactly the sort of thing that ends up in
    a commit.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    total = 0
    say(f"  fetching {url}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                digest.update(block)
                total += len(block)
    except Exception as exc:
        raise PinError(f"{url} could not be fetched ({exc})") from None
    if total == 0:
        raise PinError(f"{url} answered with nothing")
    return total, digest.hexdigest()


def pin(manifest: dict, pins: dict, check_only: bool, say) -> tuple[int, list[str]]:
    """Fill in what is missing. Returns ``(changed, disagreements)``.

    Writes into ``pins`` -- the overlay -- rather than into ``manifest``. The
    manifest is a tracked file, and a tracked file edited in place turns every
    later ``git pull`` into a merge conflict in a file full of hashes, which is
    the worst possible place to resolve one by picking a side.
    """
    changed = 0
    disagreements: list[str] = []

    for identifier, entry in (manifest.get("models") or {}).items():
        say(f"{identifier}")
        for artifact in entry.get("files") or []:
            url = artifact.get("url")
            name = artifact.get("filename", url)
            if not url:
                raise PinError(f"{identifier}: {name} has no URL")

            committed = (artifact.get("sha256") or "").strip().casefold()
            recorded = ((pins.get(name) or {}).get("sha256") or "").strip().casefold()
            size, digest = measure(url, say)

            if committed and committed != digest:
                # Reported, never overwritten. See the module docstring.
                disagreements.append(
                    f"{identifier}/{name}: checked in {committed}, published {digest}")
                continue
            if committed:
                say(f"  {name} is pinned in the manifest and still matches")
                continue
            if recorded and recorded != digest:
                disagreements.append(
                    f"{identifier}/{name}: pinned locally as {recorded}, published {digest}")
                continue
            if recorded == digest and (pins.get(name) or {}).get("bytes") == size:
                say(f"  {name} is already pinned locally and still matches")
                continue

            say(f"  {name}: {size} bytes, {digest}")
            if not check_only:
                pins[name] = {"sha256": digest, "bytes": size}
            changed += 1
    return changed, disagreements


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report what would be pinned and change nothing")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--overlay", type=Path, default=None,
                        help="where to write the pins (default: beside the manifest)")
    arguments = parser.parse_args(argv)

    def say(text):
        print(text, flush=True)

    try:
        manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"{arguments.manifest} could not be read ({exc})", file=sys.stderr)
        return 2

    overlay = arguments.overlay or arguments.manifest.with_name(OVERLAY.name)
    try:
        pins = (json.loads(overlay.read_text(encoding="utf-8")).get("artifacts") or {}
                if overlay.exists() else {})
    except (OSError, ValueError) as exc:
        print(f"{overlay} could not be read ({exc})", file=sys.stderr)
        return 2

    try:
        changed, disagreements = pin(manifest, pins, arguments.check, say)
    except PinError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    if disagreements:
        print("\nThe published artifacts no longer match what is checked in:",
              file=sys.stderr)
        for line in disagreements:
            print(f"  {line}", file=sys.stderr)
        print("\nNothing was written. A publisher who has re-uploaded is a review "
              "decision, not a script's.", file=sys.stderr)
        return 2

    if arguments.check:
        print(f"\n{changed} artifact(s) would be pinned.")
        return 1 if changed else 0

    if changed:
        overlay.write_text(json.dumps({"schema": 1, "artifacts": pins}, indent=2) + "\n",
                           encoding="utf-8")
        print(f"\nPinned {changed} artifact(s) into {overlay}.")
        print("That file is not tracked by git, so it survives a pull and never "
              "conflicts. Voice Chat reads it on the next WebUI start.")
    else:
        print("\nEverything is already pinned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
