"""TEMPORARY: the Generate box watcher's log route.

``javascript/model_chain_watch_generate.js`` watches Txt2Img's Generate box --
reported to end up back in the left column after a sequence of focus mode and
the flyout's docked settings column, with no script known to put it there --
and posts what it sees here: every DOM call that moves the box, with the
caller's stack, every change of its ancestry, and the moments around them.
This writes one line per record into the extension's own log
(``<LLM data root>/logs/model_chain.log``, see :mod:`mc_logfile`), where the
stack names the mover.

What reaches the log is built here out of a fixed set of keys: every value is
coerced to a bool, an int, or printable ASCII cut to a length, so a payload
from anywhere produces a line of this module's own shape. Elements are named
by id and class on the browser side; no text content travels.

To be removed, with the script and its registration in
``scripts/model_chain.py``, once the log has caught the mover.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory; the file by mc_logfile."""

ROUTE = "/model-chain/generate-box/watch"
"""Where the browser posts: a POST under the extension's own name."""

PREFIX = "Model Chain watch:"

FIELDS = ("box", "from", "to", "before", "after", "why", "was", "url", "fn_index",
          "trigger_id", "message", "userAgent")
"""The record's own fields, in the order they are written."""

STATE = ("focus", "docked", "open", "full", "hidden", "width", "height", "tab",
         "active", "txt2img_generate_box", "img2img_generate_box", "error")
"""The state snapshot's keys, in the order they are written."""

LIMITS = {"kind": 48, "field": 400, "state": 400, "stack_line": 300}
STACK_LINES = 20
RECORDS = 200
"""Caps: a record's kind, each field, each state value, each stack line; the
stack's lines and the records in one post."""


def _text(value, limit: int) -> str:
    """Printable ASCII only, cut to ``limit``; anything else is a ``?``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        value = str(type(value).__name__)
    cleaned = "".join(ch if 0x20 <= ord(ch) < 0x7F else "?" for ch in value)
    return cleaned[:limit]


def line_of(record) -> str:
    """One log line for one record. Never raises."""
    if not isinstance(record, dict):
        record = {}
    parts = [f"#{_text(record.get('seq'), 12) or '?'}",
             f"t={_text(record.get('t'), 20) or '?'}",
             _text(record.get("kind"), LIMITS["kind"]) or "?"]
    for name in FIELDS:
        if name in record and record.get(name) not in (None, ""):
            parts.append(f"{name}={_text(record.get(name), LIMITS['field'])}")
    state = record.get("state")
    if isinstance(state, dict):
        said = []
        for name in STATE:
            if name in state:
                said.append(f"{name}={_text(state.get(name), LIMITS['state'])}")
        if said:
            parts.append("state{" + " ".join(said) + "}")
    stack = record.get("stack")
    if isinstance(stack, list) and stack:
        lines = [_text(entry, LIMITS["stack_line"]) for entry in stack[:STACK_LINES]]
        parts.append("stack[" + " | ".join(entry for entry in lines if entry) + "]")
    return f"{PREFIX} " + " ".join(parts)


def _records(payload) -> list:
    """The records in a post: its ``records`` list, or the post itself as one
    record when it has no such key; nothing for anything else."""
    if not isinstance(payload, dict):
        return []
    if "records" in payload:
        records = payload.get("records")
        return records[:RECORDS] if isinstance(records, list) else []
    return [payload]


def lines_of(payload) -> list[str]:
    """Every record in a post, as log lines; at most :data:`RECORDS`."""
    return [line_of(record) for record in _records(payload)]


def note(payload) -> list[str]:
    """Log a post's records, a line each, and hand the lines back.

    A move or a replacement of the box is a warning, so it stands out; the
    rest -- the moments around it -- are ordinary lines.
    """
    written = []
    for record in _records(payload):
        line = line_of(record)
        kind = record.get("kind") if isinstance(record, dict) else ""
        loud = isinstance(kind, str) and (kind.startswith("dom.") or kind in ("moved", "replaced"))
        if loud:
            logger.warning(line)
        else:
            logger.info(line)
        written.append(line)
    return written


def install(_demo=None, app=None) -> bool:
    """Register :data:`ROUTE` on the WebUI's FastAPI app. Never fatal.

    Signature is ``script_callbacks.on_app_started``'s; a UI reload calls it
    again, and the route is added once.
    """
    if app is None or not hasattr(app, "add_api_route"):
        return False

    for existing in getattr(app, "routes", []):
        if getattr(existing, "path", None) == ROUTE:
            return True

    def generate_box_watch(payload: dict):
        try:
            count = len(note(payload))
        except Exception:
            logger.debug("Model Chain: could not log a Generate box watch post",
                         exc_info=True)
            count = 0
        return {"logged": count}

    try:
        app.add_api_route(ROUTE, generate_box_watch, methods=["POST"])
    except Exception:
        logger.debug("Model Chain: could not register the Generate box watch route",
                     exc_info=True)
        return False
    return True
