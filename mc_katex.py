"""The vendored KaTeX, served to the Forge Assistant's panel when it needs it.

LLM Studio's transcript typesets math with the KaTeX Gradio bundles; the
assistant's panel draws its own messages (``javascript/
forge_assistant_markdown.js``) and has no way to reach Gradio's copy -- it is a
chunk with a hashed name and no exports. So the panel has its own: the same
version, 0.16.10, in ``vendor/katex`` (MIT, ``vendor/katex/LICENSE``), fetched
the first time a message in the panel holds math, and not before. A page with no
math loads none of it.

One route, ``GET /model-chain/katex/0.16.10/<file>``, and it serves exactly the
files ``vendor/katex/manifest.json`` lists with a ``.js``, ``.css`` or
``.woff2`` name -- the script, its stylesheet and the twenty fonts the
stylesheet names relative to itself -- and answers 404 to anything else, a path
that climbs out of the folder included, without looking at the disk. The
version is in the address, so the answer can be kept by the browser for a year;
a new version is a new address.

No key, unlike the conversation's routes: what this serves is the same open
source file for everybody, and the key exists to keep a cross-site page from
reading somebody's conversation.

Not under ``from __future__ import annotations``: FastAPI reads a handler's
annotations against this module's globals (the Gallery Quadrant's 422, CLAUDE.md),
and the one here is ``str``.
"""

import json
import logging
from pathlib import Path

try:
    from fastapi.responses import Response  # type: ignore
except Exception:  # pragma: no cover - a host without FastAPI
    Response = None  # type: ignore

logger = logging.getLogger("model_chain")

VERSION = "0.16.10"
"""Also in javascript/forge_assistant_markdown.js (``KATEX_VERSION``); a test
holds the two equal."""

ROOT = Path(__file__).resolve().parent / "vendor" / "katex"
ROUTE = f"/model-chain/katex/{VERSION}/{{name:path}}"
PREFIX = f"/model-chain/katex/{VERSION}/"

TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
}

CACHE = "public, max-age=31536000, immutable"

_held: dict = {}


def served() -> dict:
    """The files the route serves, by name, from the manifest: ``{name: sha256}``."""
    try:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Model Chain: the vendored KaTeX's manifest could not be read; "
                       "the assistant's panel shows math as written")
        return {}
    listed = manifest.get("sha256") or {}
    return {name: digest for name, digest in listed.items()
            if Path(name).suffix in TYPES}


def file_for(name) -> tuple | None:
    """``(bytes, media type)`` for a served file, or None for anything else."""
    name = str(name or "")
    if name in _held:
        return _held[name]
    if name not in served():
        return None
    try:
        body = (ROOT / name).read_bytes()
    except OSError:
        return None
    found = (body, TYPES[Path(name).suffix])
    _held[name] = found
    return found


def install(_demo=None, app=None) -> bool:
    """Register :data:`ROUTE` on the WebUI's FastAPI app. Never fatal, and
    idempotent: a UI reload calls it again."""
    if app is None or not hasattr(app, "add_api_route"):
        return False
    for existing in getattr(app, "routes", []):
        if getattr(existing, "path", None) == ROUTE:
            return True
    if Response is None:
        logger.debug("Model Chain: there is no FastAPI to serve the vendored KaTeX from")
        return False

    def katex_file(name: str):
        found = file_for(name)
        if found is None:
            return Response(status_code=404, headers={"Cache-Control": "no-store"})
        body, media = found
        return Response(content=body, media_type=media,
                        headers={"Cache-Control": CACHE, "X-Content-Type-Options": "nosniff"})

    try:
        app.add_api_route(ROUTE, katex_file, methods=["GET"])
    except Exception:
        logger.debug("Model Chain: could not register the KaTeX route", exc_info=True)
        return False
    return True
