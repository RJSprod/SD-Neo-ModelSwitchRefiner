"""Extra networks: newest created first, and "created" meaning created.

The user's word: "default sort to newest created to oldest". Forge has the
settings for the order itself -- Settings, Extra Networks, *Default sorting
method for Extra Networks cards* and *Default sorting order for Extra
Networks cards*, Path and Ascending out of the box -- but not for what
"created" means. Its sort keys, unchanged since A1111, put a file's
modification time under ``date_created`` and its ``st_ctime`` under
``date_modified`` (``ExtraNetworksPage.get_sort_keys``), and on Windows
``st_ctime`` is the creation time: *Sort by date created* sorted by when a
file was last written, *Sort by date modified* by when it arrived. A LoRA
copied in from another drive keeps its old modification time, so newest
created first by Forge's key put it among the oldest.

Two things, both before Forge renders a page (``on_before_ui``):

- :func:`install` wraps ``get_sort_keys`` so that ``date_created`` is the
  file's creation time where the platform keeps one -- ``st_birthtime``
  (Windows on Python 3.12 and later, macOS), or ``st_ctime`` on Windows
  before that -- and ``date_modified`` its modification time. Where no
  creation time is kept (Linux), Forge's keys stand as they are.
- :func:`default_newest` sets Forge's two settings to Date Created and
  Descending, once, and only while both are still at Forge's own defaults:
  a choice made there stands, and a choice made there afterwards is never
  undone. The record of it is a hidden option, which is what makes it run
  once.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("model_chain")

OPT_FIELD = "extra_networks_card_order_field"
"""Forge's *Default sorting method for Extra Networks cards*."""

OPT_ORDER = "extra_networks_card_order"
"""Forge's *Default sorting order for Extra Networks cards*."""

FORGE_DEFAULTS = {OPT_FIELD: "Path", OPT_ORDER: "Ascending"}
"""What Forge starts its two settings at."""

NEWEST_CREATED = {OPT_FIELD: "Date Created", OPT_ORDER: "Descending"}
"""What the user asked for."""

OPT_RECORD = "model_chain_networks_sort_record"
"""The record of the one-time default: when, what the two settings were,
and whether they were set. Hidden: read back, never meant to be changed."""

HIDDEN_SECTION = (None, "Model Chain: Extra networks (hidden)")

_MARK = "_model_chain_creation_times"


def hidden_settings(shared) -> dict:
    return {
        OPT_RECORD: shared.OptionInfo("", "The record of the one-time default sort, newest created first"),
    }


def _windows() -> bool:
    return os.name == "nt"


_stat = os.stat


def file_times(path):
    """``(created, modified)`` for a file, created None where the platform
    keeps no creation time; ``(None, None)`` when the file cannot be read.
    Not following a link, as Forge's own lister does not."""
    try:
        stat = _stat(path, follow_symlinks=False)
    except (OSError, TypeError, ValueError):
        return None, None
    created = getattr(stat, "st_birthtime", None)
    if created is None and _windows():
        created = stat.st_ctime
    return created, stat.st_mtime


def corrected(keys, path):
    """Forge's sort keys for a file, with ``date_created`` its creation time
    and ``date_modified`` its modification time wherever the creation time
    is known; otherwise the keys as Forge made them."""
    if not isinstance(keys, dict) or "date_created" not in keys:
        return keys
    created, modified = file_times(path)
    if created is None or modified is None:
        return keys
    fixed = dict(keys)
    fixed["date_created"] = int(created)
    fixed["date_modified"] = int(modified)
    return fixed


def install(module=None) -> bool:
    """Wrap Forge's ``ExtraNetworksPage.get_sort_keys``, once. Every page
    Forge ships (checkpoints, LoRA, textual inversion) builds its cards'
    sort keys through it. True when the wrapper is in place."""
    if module is None:
        try:
            from modules import ui_extra_networks as module
        except Exception:
            return False
    page = getattr(module, "ExtraNetworksPage", None)
    original = getattr(page, "get_sort_keys", None) if page is not None else None
    if original is None:
        return False
    if getattr(original, _MARK, False):
        return True

    def get_sort_keys(self, path):
        keys = original(self, path)
        try:
            return corrected(keys, path)
        except Exception:
            return keys

    setattr(get_sort_keys, _MARK, True)
    get_sort_keys.__wrapped__ = original
    get_sort_keys.__doc__ = original.__doc__
    page.get_sort_keys = get_sort_keys
    return True


def default_newest(opts, *, save=None, now=None) -> dict | None:
    """Forge's default sort set to newest created first, once. Returns the
    record, or None when there was nothing to do: already done, or a Forge
    without these two settings (nothing is recorded then, so a Forge that
    gains them gets it)."""
    data = getattr(opts, "data", None)
    labels = getattr(opts, "data_labels", None)
    if not isinstance(data, dict) or not isinstance(labels, dict):
        return None
    if data.get(OPT_RECORD):
        return None
    if any(name not in labels for name in FORGE_DEFAULTS):
        return None
    was = {name: data.get(name, getattr(labels[name], "default", None)) for name in FORGE_DEFAULTS}
    untouched = was == FORGE_DEFAULTS
    if untouched:
        for name, value in NEWEST_CREATED.items():
            opts.set(name, value)
    record = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now() if now else time.time())),
        "was": was,
        "set": untouched,
    }
    opts.set(OPT_RECORD, json.dumps(record, ensure_ascii=False))
    if save is not None:
        try:
            save()
        except Exception:
            logger.warning("Model Chain: the extra networks' default sort could not be saved", exc_info=True)
    if untouched:
        logger.info("Model Chain: extra networks sort newest created first by default "
                    "(Settings, Extra Networks: Date Created, Descending)")
    return record


def on_before_ui():
    """The host's hook before it builds the page: Forge renders the extra
    network pages while it builds it, with the sort keys and the default
    sort as they are then."""
    try:
        from modules import shared
    except Exception:
        return
    try:
        install()
    except Exception:
        logger.warning("Model Chain: could not correct the extra networks' creation times", exc_info=True)
    try:
        default_newest(shared.opts, save=lambda: shared.opts.save(shared.config_filename))
    except Exception:
        logger.warning("Model Chain: could not set the extra networks' default sort", exc_info=True)
