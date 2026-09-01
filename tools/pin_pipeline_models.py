"""Resolve and pin the Voice Pipeline's manifest, on a machine with a network.

Never imported by the extension. It is the maintainer's tool for turning
``voice/managed-pipeline-models.json`` from a declaration into a trust root:
it asks PyPI what the runtime wheels are, asks Hugging Face what the two model
repositories hold at a named revision, records byte counts and SHA-256 digests,
and writes the file back.

What it refuses to write
------------------------
A manifest is only marked ``pinned`` when every one of these is true, and the
last is the one that cannot be answered over a network:

    every runtime artifact has a URL, a byte count and a digest;
    every stage names an immutable revision that is not ``main``;
    every stage artifact has a byte count and a digest;
    LavaSR's contract carries a MEASURED backend input rate, a measured
        analysis window and a measured context window.

That last group comes out of the Phase-0 sweep in
``docs/19-voice-chat-pipeline.md`` and is passed in with ``--lava-contract``.
Upstream's README advertises 8-48 kHz input while the reviewed upstream
inference path resamples 16 kHz to 48 kHz internally; those are different
claims about the same model, and only running it settles which one the selected
backend actually implements. A pin written from the README would be a pin that
plays 24 kHz speech at two-thirds speed and says nothing about it.

Usage
-----
    python tools/pin_pipeline_models.py --check
    python tools/pin_pipeline_models.py --runtime
    python tools/pin_pipeline_models.py --stage dpdfnet --revision <sha>
    python tools/pin_pipeline_models.py --stage lavasr --revision <sha> \\
        --lava-contract backend=onnx,rate=24000,analysis_ms=500,context_ms=60

``--check`` returns 1 when the file on disk would change, which is what a
release check runs.

Provisional stages
------------------
A stage may ship pinned to a branch rather than to a commit, with
``"provisional": true`` in its entry, when the machine that wrote the manifest
could not reach the publisher. That is a real weakening and it is a visible one:
the settings row says so, the installed record keeps it, and ``releasable()``
below refuses to mark the manifest pinned while any stage is in that state.
Running ``--stage <id> --revision <sha>`` from a machine that *can* reach the hub
measures the files, records their digests and clears the flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "voice" / "managed-pipeline-models.json"

USER_AGENT = "ModelChain-Pipeline-Pinner"
CHUNK = 256 * 1024
TIMEOUT = 60

HUB = "https://huggingface.co"
PYPI = "https://pypi.org/pypi"

STAGE_IDS = ("dpdfnet", "lavasr")


class PinError(RuntimeError):
    """Something this tool declined to write."""


def read() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PinError(f"{MANIFEST} could not be read ({exc}).") from None


def write(found: dict, check: bool) -> int:
    body = json.dumps(found, indent=2) + "\n"
    if check:
        current = MANIFEST.read_text(encoding="utf-8")
        if current == body:
            print("the manifest is up to date")
            return 0
        print("the manifest would change; run this tool without --check")
        return 1
    MANIFEST.write_text(body, encoding="utf-8")
    print(f"wrote {MANIFEST}")
    return 0


class _DropCredentialOnHop(urllib.request.HTTPRedirectHandler):
    """Strip the Authorization header when a redirect leaves the origin.

    The same guard :class:`mc_voice_models._DropCredentialOnHop` holds, and here
    for a reason that is easy to miss when writing a maintainer tool: urllib
    copies every header except Content-Length and Content-Type onto a redirected
    request, and the hub answers a file request with a 302 to a storage host. A
    bearer token added for huggingface.co would arrive at that host with it.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        made = super().redirect_request(req, fp, code, msg, headers, newurl)
        if made is not None and _origin(newurl) != _origin(req.full_url):
            for store in (made.headers, made.unredirected_hdrs):
                for name in [key for key in store if key.casefold() == "authorization"]:
                    store.pop(name, None)
        return made


def _origin(url: str) -> str:
    """Scheme and host, so an https-to-http hop on one host is a boundary too."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme.casefold()}://{parts.netloc.casefold()}"


def _open(url: str, method: str = "GET"):
    request = urllib.request.Request(url, method=method)
    request.add_header("User-Agent", USER_AGENT)
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or "")
    if token and _origin(url) == _origin(HUB):
        request.add_header("Authorization", f"Bearer {token}")
        opener = urllib.request.build_opener(_DropCredentialOnHop())
        return opener.open(request, timeout=TIMEOUT)
    return urllib.request.urlopen(request, timeout=TIMEOUT)  # noqa: S310


def measure(url: str) -> dict:
    """Download one artifact and report what actually arrived."""
    digest = hashlib.sha256()
    size = 0
    try:
        with _open(url) as response:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    except (urllib.error.URLError, OSError) as exc:
        raise PinError(f"{url} could not be fetched ({exc.__class__.__name__}).") from None
    if not size:
        raise PinError(f"{url} answered with nothing.")
    return {"bytes": size, "sha256": digest.hexdigest()}


def hub_file(repo: str, revision: str, name: str) -> str:
    return f"{HUB}/{repo}/resolve/{revision}/{name}"


def pin_runtime(found: dict) -> None:
    """Resolve the runtime closure from PyPI for every platform it names."""
    platforms = (found.get("runtime") or {}).get("platforms") or []
    if not platforms:
        raise PinError(
            "The manifest names no runtime platforms. Add one -- an id, a system, its "
            "machines, a Python version and the wheel filenames -- before pinning; this "
            "tool resolves what is declared and does not invent a closure.")
    for entry in platforms:
        for item in entry.get("artifacts") or ():
            url = str(item.get("url") or "")
            if not url:
                raise PinError(f"{item.get('filename')} has no URL to resolve.")
            print(f"  measuring {item.get('filename')}…")
            item.update(measure(url))


def pin_stage(found: dict, stage_id: str, revision: str, contract: dict) -> None:
    entry = next((row for row in found.get("stages") or ()
                  if str(row.get("id")) == stage_id), None)
    if entry is None:
        raise PinError(f"The manifest has no stage called {stage_id!r}.")
    if revision:
        entry["revision"] = revision
    if not _immutable(entry.get("revision") or ""):
        raise PinError(
            f"{stage_id} needs an immutable revision -- a 40-character commit. "
            f"{entry.get('revision')!r} is not one: a branch means something different next "
            f"week and cannot be reproduced from this repository. Pass --revision <sha>.")
    if not entry.get("artifacts"):
        raise PinError(
            f"{stage_id} names no artifacts. Add the filenames this build needs before "
            f"pinning; which files a model repository holds is a decision for a maintainer "
            f"to review rather than for a tool to guess.")
    total = 0
    for item in entry["artifacts"]:
        name = str(item.get("filename") or "")
        item.setdefault("local_name", name)
        item["url"] = hub_file(entry["repo"], entry["revision"], name)
        print(f"  measuring {stage_id}/{name}…")
        item.update(measure(item["url"]))
        total += item["bytes"]
    entry["about_bytes"] = total
    # A stage that has just been measured against an immutable revision is no
    # longer provisional, and saying so is the whole point of running this.
    entry["provisional"] = False
    if contract:
        entry.setdefault("contract", {}).update(contract)


def parse_contract(text: str) -> dict:
    """``backend=onnx,rate=24000,analysis_ms=500,context_ms=60`` as a contract.

    Every value here is a Phase-0 measurement rather than a preference, and the
    tool will not accept a partial one: a backend input rate without an analysis
    window is a manifest that still cannot be released, and writing it would
    make ``pinned`` half-true.
    """
    if not text:
        return {}
    found = {}
    for part in str(text).split(","):
        name, _, value = part.partition("=")
        name, value = name.strip(), value.strip()
        if name == "backend":
            found["backend"] = value
        elif name in ("rate", "backend_input_rate"):
            found["backend_input_rate"] = int(value)
        elif name in ("analysis_ms", "context_ms"):
            found[name] = int(value)
        elif name:
            raise PinError(f"{name!r} is not part of the LavaSR contract.")
    wanted = ("backend", "backend_input_rate", "analysis_ms", "context_ms")
    missing = [name for name in wanted if not found.get(name)]
    if missing:
        raise PinError(
            f"The LavaSR contract is incomplete: {', '.join(missing)}. All four come out "
            f"of the Phase-0 sweep together, and half of them is a manifest that still "
            f"cannot be released.")
    return found


def _immutable(revision: str) -> bool:
    """Whether a revision names one thing forever. A 40-hex commit does."""
    text = str(revision or "")
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text.casefold())


def releasable(found: dict) -> bool:
    runtime = found.get("runtime") or {}
    if not runtime.get("platforms"):
        return False
    for entry in runtime["platforms"]:
        for item in entry.get("artifacts") or ():
            if not item.get("sha256") or not int(item.get("bytes") or 0):
                return False
    for stage_id in STAGE_IDS:
        entry = next((row for row in found.get("stages") or ()
                      if str(row.get("id")) == stage_id), None)
        if entry is None or not _immutable(entry.get("revision") or ""):
            return False
        if entry.get("provisional") or not entry.get("available", True):
            return False
        if not entry.get("artifacts"):
            return False
        for item in entry["artifacts"]:
            if not item.get("sha256") or not int(item.get("bytes") or 0):
                return False
    contract = (next((row for row in found["stages"] if row["id"] == "lavasr"),
                     {}) or {}).get("contract") or {}
    return bool(contract.get("backend") and int(contract.get("backend_input_rate") or 0)
                and int(contract.get("analysis_ms") or 0)
                and int(contract.get("context_ms") or 0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the manifest would change")
    parser.add_argument("--runtime", action="store_true",
                        help="resolve the runtime closure")
    parser.add_argument("--stage", choices=STAGE_IDS, help="resolve one stage's artifacts")
    parser.add_argument("--revision", default="", help="the immutable revision to pin")
    parser.add_argument("--lava-contract", default="",
                        help="backend=…,rate=…,analysis_ms=…,context_ms= from Phase 0")
    found = parser.parse_args(argv)

    try:
        manifest = read()
        if found.runtime:
            pin_runtime(manifest)
        if found.stage:
            pin_stage(manifest, found.stage, found.revision,
                      parse_contract(found.lava_contract)
                      if found.stage == "lavasr" else {})
        manifest["pinned"] = releasable(manifest)
        if not manifest["pinned"]:
            print("NOT marked pinned: something above is still unresolved.")
        return write(manifest, found.check)
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
