"""Fill in the immutable revision and exact byte count for every catalogue entry.

Not part of the extension and never imported by it. This is a maintainer's tool,
run once when the catalogue changes, on a machine that can reach
huggingface.co -- which is why it lives in ``tools/`` rather than in ``scripts/``
where Forge would import it at startup.

What it is for
--------------
``prompt_master/models/managed-models.json`` ships with ``"revision": "main"``
and ``"bytes": null`` for entries nobody has pinned yet. That is safe -- every
downloaded byte is checked against the SHA-256 committed in this repository, so
a publisher who re-uploads gets a refusal rather than a substitution -- but it
is not *good*: a moved branch turns into a hash mismatch and a sentence asking
for an extension update, where a pinned commit would simply have kept working.
This turns the first into the second.

What it will not do
-------------------
It never invents or changes a SHA-256. The hash in the repository is the trust
root for the whole feature, so this tool *checks* what the hub reports against
what is checked in and refuses the whole run on any disagreement, rather than
writing the hub's answer over ours. A publisher whose files really have changed
is a review decision -- somebody has to look at the new model card and decide --
and not something a script gets to do at three in the morning.

    python tools/pin_managed_models.py             # write the pins
    python tools/pin_managed_models.py --check     # report, change nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY = (Path(__file__).resolve().parent.parent
            / "prompt_master" / "models" / "managed-models.json")

API = "https://huggingface.co/api/models/{repo_id}/revision/{revision}"
RESOLVE = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"

TIMEOUT = 30.0


class PinError(RuntimeError):
    """Something a maintainer has to look at rather than re-run."""


@dataclass
class Resolved:
    """One repository at one commit, and what its files really are."""

    commit: str
    files: dict = field(default_factory=dict)
    """``{filename: (size_bytes, sha256 or "")}``. The hash is empty when the
    hub did not report one, which is a reason to pin the size alone rather than
    a reason to fail: the download still verifies against what is checked in."""


def resolve(repo_id: str, revision: str, filenames, fetch) -> Resolved:
    """Ask the hub for the commit ``revision`` points at, and for each file.

    ``fetch`` is injected so the parsing above it can be tested without a
    network: it takes ``(method, url)`` and returns ``(status, headers, body)``.
    """
    status, _headers, body = fetch("GET", API.format(repo_id=repo_id, revision=revision))
    if status != 200:
        raise PinError(f"{repo_id}: the hub answered {status} for revision {revision}")
    try:
        commit = str(json.loads(body)["sha"])
    except (ValueError, KeyError, TypeError) as exc:
        raise PinError(f"{repo_id}: the hub's revision answer has no commit sha ({exc})") from None
    if len(commit) != 40:
        raise PinError(f"{repo_id}: {commit!r} is not a commit sha")

    found = Resolved(commit)
    for filename in filenames:
        status, headers, _body = fetch(
            "HEAD", RESOLVE.format(repo_id=repo_id, revision=commit, filename=filename))
        if status >= 400:
            raise PinError(f"{repo_id}: the hub answered {status} for {filename} at {commit}")
        found.files[filename] = (_size(headers), _sha256(headers))
    return found


def _header(headers: dict, name: str) -> str:
    """One header, matched without regard to case, which HTTP does not have."""
    folded = {str(key).casefold(): value for key, value in (headers or {}).items()}
    return str(folded.get(name.casefold(), "") or "")


def _size(headers: dict) -> int:
    """The real size of the file.

    ``x-linked-size`` first: a large model on the hub is an LFS object, and a
    plain ``content-length`` on the pointer would be a few hundred bytes -- a
    number that looks like an answer and is not one.
    """
    for name in ("x-linked-size", "content-length"):
        raw = _header(headers, name)
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 0


def _sha256(headers: dict) -> str:
    """The hub's own SHA-256 for the file, when it reports one.

    For an LFS object the etag is the object's sha256, sometimes bare and
    sometimes prefixed. Anything else -- a weak etag, an md5, a hash of a
    pointer file -- is not a sha256 and is reported as absent rather than
    guessed at.
    """
    for name in ("x-linked-etag", "etag"):
        raw = _header(headers, name).strip().strip('"').removeprefix("sha256:")
        if len(raw) == 64:
            try:
                int(raw, 16)
            except ValueError:
                continue
            return raw.casefold()
    return ""


def pin(document: dict, resolver) -> tuple[dict, list[str]]:
    """Return ``document`` with pins filled in, and a line per change.

    ``resolver`` takes ``(repo_id, revision, filenames)`` and returns a
    :class:`Resolved`. Raises :class:`PinError` -- before writing anything --
    if the hub reports a hash that disagrees with the one checked in.
    """
    updated = json.loads(json.dumps(document))
    changes: list[str] = []
    for model in updated.get("models") or []:
        artifacts = [model[key] for key in ("model", "projector") if model.get(key)]
        names = [artifact["filename"] for artifact in artifacts]
        found = resolver(model["repo_id"], model["revision"], names)

        for artifact in artifacts:
            size, sha256 = found.files.get(artifact["filename"], (0, ""))
            committed = str(artifact.get("sha256") or "").casefold()
            if sha256 and sha256 != committed:
                raise PinError(
                    f"{model['id']}: the hub says {artifact['filename']} is {sha256}, and this "
                    f"repository says {committed}. The published artifact has changed — review "
                    f"the model card and update the registry by hand. Nothing was written."
                )
            if size and artifact.get("bytes") != size:
                changes.append(f"{model['id']}: {artifact['filename']} is {size} bytes")
                artifact["bytes"] = size

        if model["revision"] != found.commit:
            changes.append(f"{model['id']}: pinned to {found.commit}")
            model["revision"] = found.commit
    return updated, changes


def _httpx_fetch(method: str, url: str):
    import httpx

    with httpx.Client(follow_redirects=True, timeout=TIMEOUT) as client:
        response = client.request(method, url)
        return response.status_code, dict(response.headers), response.text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--check", action="store_true",
                        help="report what would change and write nothing")
    options = parser.parse_args(argv)

    document = json.loads(options.registry.read_text(encoding="utf-8"))
    try:
        updated, changes = pin(
            document,
            lambda repo_id, revision, names: resolve(repo_id, revision, names, _httpx_fetch))
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # A proxy, a firewall or an offline machine. The registry is unchanged
        # either way, and a traceback would not say that.
        print(f"error: huggingface.co could not be reached ({exc.__class__.__name__}: {exc}). "
              f"{options.registry} is unchanged.", file=sys.stderr)
        return 2

    if not changes:
        print("Every entry is already pinned and sized.")
        return 0
    for line in changes:
        print(line)
    if options.check:
        print(f"\n{len(changes)} change(s) — nothing written (--check).")
        return 1
    options.registry.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(changes)} change(s) to {options.registry}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
