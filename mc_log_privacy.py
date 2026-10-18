"""Nothing this extension logs names a path on this machine or a file of the user's.

The rule, the user's and now a standing convention of this repository: a log
line never carries a path on this machine, the name of a file or folder the
user made or chose -- an image, a render, a sample, an attachment, a
conversation, the folder a save goes to -- or the user's own text. A log is
shared when something goes wrong, and what it says about the machine it came
from should be nothing. Models and LoRAs are named as the host names them,
because a log that cannot say which model was loaded explains nothing;
presets, profiles, layouts and characters are named, because they are choices
from a list; counts, sizes and durations are what the lines are made of.

Two layers hold it. The lines themselves: every logger call in the extension
was read and the ones that carried a path or a user's file were rewritten
(``tests/test_log_privacy.py`` holds the ones that used to). And this filter,
on the ``model_chain`` logger every module logs through: any absolute path
that reaches a record anyway -- in the message, in an argument, in a
traceback -- is replaced by ``<path>`` before any handler sees it, so a line
written next year cannot leak one either. The placeholder keeps the sentence
readable and says that something was removed.

A path is recognised by its shape, not by a list of this machine's folders:
a drive letter and a separator (``C:\\...``), a UNC share (``\\\\host\\...``),
a POSIX path under one of the roots a user's files or an installation live
in (``/home``, ``/Users``, ``/mnt``, ...), a home-relative path (``~/...``)
and a ``file:`` URL. A route of this extension's own (``/model-chain/...``)
is not one, a checkpoint's name under the models folder (``Krea\\x.safetensors``)
is not one, and a clock (``13:35:43``) is not one.
"""

from __future__ import annotations

import logging
import re
import traceback

PLACEHOLDER = "<path>"

_TAIL = ".,;:!?)]}"
"""Punctuation a path at the end of a sentence is followed by; given back."""

_PATTERNS = (
    # A file URL, first: its path would otherwise be taken by the pattern
    # below and leave the scheme standing in front of the placeholder.
    re.compile(r"\bfile:/{2,3}[^\s\"'<>|]*"),
    # A drive letter, a separator, then anything up to a space or a quote.
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|*?]*"),
    # A UNC share.
    re.compile(r"(?<![\w\\])\\\\[^\s\\\"'<>|]+\\[^\s\"'<>|]*"),
    # A POSIX path under a root a user's files or an installation live in.
    re.compile(
        r"(?<![\w./:\\-])/(?:home|Users|root|mnt|media|tmp|var|opt|srv|data|usr|etc|run|"
        r"private|Volumes|workspace|content)(?=$|[/\s\"'<>|.,;:!?)\]}])(?:/[^\s\"'<>|]*)?"
    ),
    # A home-relative path.
    re.compile(r"(?<![\w])~[\\/][^\s\"'<>|]*"),
)


def _swap(match: re.Match) -> str:
    found = match.group(0)
    tail = ""
    while found and found[-1] in _TAIL:
        tail = found[-1] + tail
        found = found[:-1]
    return PLACEHOLDER + tail


def redact(text: str) -> str:
    """``text`` with every path in it replaced by :data:`PLACEHOLDER`."""
    if not text:
        return text
    for pattern in _PATTERNS:
        text = pattern.sub(_swap, text)
    return text


def _format_exception(exc_info) -> str:
    return "".join(traceback.format_exception(*exc_info)).rstrip("\n")


class NoPaths(logging.Filter):
    """The filter on the logger. Never refuses a record; rewrites it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            cleaned = redact(message)
            if cleaned != message or record.args:
                record.msg = cleaned
                record.args = ()
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
            elif record.exc_info and record.exc_info != (None, None, None):
                # Cached on the record, which is what every handler's formatter
                # reads before it would format the traceback itself.
                record.exc_text = redact(_format_exception(record.exc_info))
            if record.stack_info:
                record.stack_info = redact(record.stack_info)
        except Exception:
            # A filter that raises takes the caller with it, usually in the
            # middle of a generation. An unredacted line is the lesser harm.
            pass
        return True


def installed(logger: logging.Logger) -> bool:
    return any(isinstance(existing, NoPaths) for existing in logger.filters)


def install(logger: logging.Logger) -> None:
    """Put the filter on ``logger`` once."""
    if not installed(logger):
        logger.addFilter(NoPaths())
