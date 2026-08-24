"""This extension's own log, written to a file next to the LLM's.

Every module here logs through ``logging.getLogger("model_chain")``, and
:func:`mc_memory._make_logger` hands that logger the host's console handler --
so until now every line this extension wrote existed only in the terminal Forge
was started from. Close the window, or run it as a service, and the reason a
switch was slow went with it.

The obvious place for a copy is the folder the extension already keeps things
in: ``<LLM data root>/logs``, beside the ``llama-server.log`` the managed
runtime writes. Nothing new to find, one folder for both.

    <WebUI data directory>/model_chain_llm/logs/
        llama-server.log      the managed LLM's own output
        model_chain.log       this

Attached on ``on_app_started``, which is the first moment the settings that
decide where that folder *is* are certainly loaded -- ``model_chain_llm_root``
can move it, and a log file written to the wrong place would be worse than no
log file at all. Lines from before that point are still the console's alone; the
first line in the file names the file, so a "where did my logs go" question
answers itself from either end.

Rotating, because a log nobody prunes is a disk somebody loses: 2 MB and three
old copies, which is a long generation history and about the size of one photo.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory. This adds a second one."""

FILENAME = "model_chain.log"

MAX_BYTES = 2 * 1024 * 1024
BACKUPS = 3
"""2 MB before it rolls, three kept. See the module docstring."""

_MARK = "_model_chain_log_file"
"""Set on the handler this module owns, so a second call finds it.

By attribute and not by class or filename: ``on_app_started`` can fire more than
once across a UI reload, and a second handler on the same file is every line
written twice.
"""


def directory() -> Path | None:
    """``<LLM data root>/logs``, or None if that cannot be worked out."""
    try:
        import mc_llm_paths

        return Path(mc_llm_paths.app_paths().logs)
    except Exception:
        logger.debug("Model Chain: could not work out where the log folder is",
                     exc_info=True)
        return None


def path() -> Path | None:
    """The log file this module writes, or None."""
    folder = directory()
    return None if folder is None else folder / FILENAME


def attached() -> logging.Handler | None:
    """The handler this module attached, if it is on the logger."""
    for handler in list(logger.handlers):
        if getattr(handler, _MARK, False):
            return handler
    return None


def attach(_demo=None, _app=None) -> bool:
    """Start writing the extension's log to :func:`path`. Never fatal.

    Signature is ``script_callbacks.on_app_started``'s -- ``(demo, app)`` --
    and neither half is used: this needs the settings to be loaded, which by
    then they are, and nothing else.

    A read-only data directory, a root pointed somewhere that no longer exists,
    a build without the vendored package: each of those is a False here and a
    console log that carries on exactly as it did.
    """
    if attached() is not None:
        return True

    target = path()
    if target is None:
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            target, maxBytes=MAX_BYTES, backupCount=BACKUPS,
            encoding="utf-8", delay=True)
        # The clock and the "Model Chain" prefix are already on the message --
        # `mc_memory._Timestamped` is a filter on the *logger*, so it has run by
        # the time any handler sees the record. Only the level is missing.
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        setattr(handler, _MARK, True)
        logger.addHandler(handler)
    except Exception:
        logger.debug("Model Chain: could not open the log file", exc_info=True)
        return False

    logger.info("Model Chain: this log is also being written to %s", target)
    return True
