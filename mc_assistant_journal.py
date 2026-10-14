"""The Forge Assistant's journal: what the page saw, in the extension's own log.

The flyout's drag went wrong on the user's machine twice ("it drags the whole
page") and never here. The page's account of a drag -- ``accountForDrag`` in
``javascript/forge_assistant.js``: the press, the handle's rules, what else
moved, the events as they came, the browser and the page they happened on --
went to the browser's console, which a phone has no way to show and a desktop
has to be asked for. At the user's word ("write it here, with other logging
content") it goes into the extension's own log instead, the one they already
share:

    <WebUI data directory>/model_chain_llm/logs/model_chain.log

one line per account, through the ``model_chain`` logger like every other
line (:mod:`mc_logfile` writes that logger to the file; the console gets it
too), so a drag sits in the log between the lines of whatever else was going
on at the time. The route (``POST /model-chain/conversation/v2/journal``,
behind the same key as every conversation route) takes the account as the
page sends it. Nothing of it is read into anything here, on purpose: the
point is what the browser said, not what the server made of it.
"""

from __future__ import annotations

import json
import logging
import re

import mc_logfile

logger = logging.getLogger("model_chain")

FILE = mc_logfile.FILENAME
"""Where an account lands: the extension's log, ``model_chain.log``."""

MAX_ENTRY = 64 * 1024
"""One account, serialised. The page's run to a few kilobytes; a bigger one
is not an account of a drag."""

_KIND = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
"""What an entry may call itself: ``drag``, and ``churn`` -- a page that
kept updating while nobody was doing anything, with what kept changing
(the Forge Assistant's quiet-page watch, ``watchChurn``)."""


def summary(entry: dict) -> str:
    """One clause of an account, so the line reads before its JSON does."""
    if entry.get("kind") == "churn":
        return _churn_summary(entry)
    parts = [str(entry.get("pointer") or "?") + " on the " + str(entry.get("handle") or "?")]
    travel = entry.get("travel")
    if isinstance(travel, (int, float)):
        parts.append(f"{int(travel)} px")
    ended = entry.get("ended")
    if ended:
        parts.append("ended by " + str(ended))
    astray = entry.get("astray")
    if isinstance(astray, list) and astray:
        parts.append("; ".join(str(item) for item in astray))
    else:
        parts.append("nothing else moved")
    return ", ".join(parts)


def _churn_summary(entry: dict) -> str:
    """``24 after-update runs in 7.3 s, nothing pressed for 12 s; most
    changed: #txt2img_settings 40, div.progress 12``: the page's after-update
    runs while it was idle, and the nodes whose child lists changed most --
    where to look for the callback that keeps the page updating."""
    runs = entry.get("runs")
    seconds = entry.get("seconds")
    quiet = entry.get("quiet")
    head = f"{runs if isinstance(runs, (int, float)) else '?'} after-update runs"
    if isinstance(seconds, (int, float)):
        head += f" in {seconds:g} s"
    if isinstance(quiet, (int, float)):
        head += f", nothing pressed for {quiet:g} s"
    targets = entry.get("targets")
    if isinstance(targets, list) and targets:
        named = [f"{item.get('node')} {item.get('count')}" for item in targets if isinstance(item, dict)]
        if named:
            head += "; most changed: " + ", ".join(named)
    return head


def line(entry: dict) -> str:
    """The log line for one account. Raises ValueError for what is not one."""
    if not isinstance(entry, dict):
        raise ValueError("An account is an object.")
    kind = entry.get("kind", "drag")
    if not isinstance(kind, str) or not _KIND.match(kind):
        raise ValueError("An account says what kind it is, in one word.")
    try:
        serialised = json.dumps({key: value for key, value in entry.items() if key != "kind"},
                                ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        raise ValueError("An account is JSON.")
    if len(serialised.encode("utf-8")) > MAX_ENTRY:
        raise ValueError("That account is too large to be one.")
    return f"Model Chain: Forge Assistant {kind} ({summary(entry)}): {serialised}"


def record(entry: dict) -> str:
    """Write one account to the log. Returns the file's name. Raises
    ValueError for what is not an account; never for the log itself, which
    is the console's as well and cannot fail to be the console's."""
    written = line(entry)
    logger.info(written)
    return FILE
