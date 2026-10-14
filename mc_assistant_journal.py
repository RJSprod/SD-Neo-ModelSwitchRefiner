"""The Forge Assistant's journal: what the page saw, in the extension's own log.

What the page writes here goes into the log the user already shares:

    <WebUI data directory>/model_chain_llm/logs/model_chain.log

one line per entry, through the ``model_chain`` logger like every other line
(:mod:`mc_logfile` writes that logger to the file; the console gets it too),
so an entry sits between the lines of whatever else was going on. The route
(``POST /model-chain/conversation/v2/journal``, behind the same key as every
conversation route) takes the entry as the page sends it. Nothing of it is
read into anything here, on purpose: the point is what the browser said, not
what the server made of it.

One kind is kept: ``churn``, from the quiet-page watch (``watchChurn`` in
``javascript/forge_assistant.js``) -- a page that kept updating while nobody
was doing anything, with the nodes whose child lists changed most.

The journal was built for another, ``drag``: every drag of the flyout wrote
an account of itself here (handoff 43, docs/22 §3.34), and that account is
how a Gecko browser was found panning the page under a finger whatever the
page said about the touch. The question answered and the hold built, the
user asked for those lines to go: "remove those flyout movement logs from all
logs, i should not see those drag events anymore". The page sends none now,
and one from a page loaded before the update -- until that page is reloaded
-- is answered and not written (:data:`RETIRED`).
"""

from __future__ import annotations

import json
import logging

import mc_logfile

logger = logging.getLogger("model_chain")

FILE = mc_logfile.FILENAME
"""Where an entry lands: the extension's log, ``model_chain.log``."""

MAX_ENTRY = 64 * 1024
"""One entry, serialised. The page's run to a kilobyte or two; a bigger one
is not one of its entries."""

KINDS = ("churn",)
"""What the journal keeps: ``churn``, the quiet-page watch's line."""

RETIRED = ("drag",)
"""What it no longer keeps: ``drag``, the flyout's account of each drag,
removed at the user's word. Answered, so a page loaded before the update has
nothing to complain of, and never written."""


def summary(entry: dict) -> str:
    """``24 after-update runs in 7.3 s, nothing pressed for 12 s; most
    changed: div#txt2img_settings 40, div.progress 12``: the page's
    after-update runs while it was idle, and the nodes whose child lists
    changed most -- where to look for the callback that keeps the page
    updating. One clause, so the line reads before its JSON does."""
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
    """The log line for one entry. Raises ValueError for what is not one."""
    if not isinstance(entry, dict):
        raise ValueError("An entry is an object.")
    kind = entry.get("kind")
    if kind not in KINDS:
        raise ValueError("The journal keeps one kind of entry: churn.")
    try:
        serialised = json.dumps({key: value for key, value in entry.items() if key != "kind"},
                                ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        raise ValueError("An entry is JSON.")
    if len(serialised.encode("utf-8")) > MAX_ENTRY:
        raise ValueError("That entry is too large to be one.")
    return f"Model Chain: Forge Assistant {kind} ({summary(entry)}): {serialised}"


def record(entry: dict) -> str | None:
    """Write one entry to the log. Returns the file's name, or None for a
    retired kind, which is answered and not written. Raises ValueError for
    what is not an entry; never for the log itself, which is the console's
    as well and cannot fail to be the console's."""
    if isinstance(entry, dict) and entry.get("kind") in RETIRED:
        return None
    written = line(entry)
    logger.info(written)
    return FILE
