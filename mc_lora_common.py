"""Small helpers shared by the LoRA details modules: the logger, a settings
reader, the host's LoRA folders, and the normalisers the CSV library and the
Civitai client agree on.

Carried in from the user's ``SD-ForgeNeo-ExtendedLoraDetails`` extension
(2026-10-23) with the rest of ``mc_lora_*`` and ``mc_extra_networks_*``; see
``docs/26-lora-details.md`` for the map. Everything here logs through the
``model_chain`` logger, so a line passes the path filter every other module's
does, and nothing here imports Gradio or the host at import time.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

TAG = "Model Chain LoRA details"
"""How these modules' lines open, the way the theme's open with its name."""

SAFETENSORS_EXTENSIONS = (".safetensors", ".safetensor")
NETWORK_EXTENSIONS = (".safetensors", ".safetensor", ".pt", ".ckpt")


def log(message: str) -> None:
    """A line of ordinary news. Never a path or a user's file in it (CLAUDE.md)."""
    logger.info("%s: %s", TAG, message)


def report(message: str, exc_info: bool = True) -> None:
    """Something went wrong but the extension carries on; the traceback goes
    with it, which the logger's filter reads for paths like any line."""
    logger.warning("%s: %s", TAG, message, exc_info=exc_info)


def opt(name: str, fallback):
    """Read a WebUI setting without exploding when the UI is not up yet.

    Settings are absent from ``opts`` during early imports and in tests, and a
    missing setting must not change behaviour, so the fallback is the
    documented default rather than a disabled one.
    """
    try:
        from modules import shared

        value = getattr(shared.opts, name, fallback)
        return fallback if value is None else value
    except Exception:
        return fallback


def lora_directories() -> list[str]:
    """Every directory Forge scans for LoRA networks."""
    try:
        from modules import shared

        dirs = [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])]
    except Exception:
        return []

    seen: list[str] = []
    for entry in dirs:
        if not entry:
            continue
        resolved = os.path.abspath(os.path.expanduser(str(entry)))
        if resolved not in seen:
            seen.append(resolved)
    return seen


def normalize_key(value) -> str:
    """Lower-case a column name / label down to ``[a-z0-9_]``."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def clean_cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


_HEX_RE = re.compile(r"^[0-9a-f]+$")


def normalize_hash(value) -> str:
    """Return a lower-case hex hash, or ``""`` when the value is not hex."""
    text = clean_cell(value).lower()
    if not text:
        return ""
    if text.startswith("0x"):
        text = text[2:]
    if len(text) < 8 or not _HEX_RE.match(text):
        return ""
    return text


def normalize_path(value) -> str:
    """Case-folded POSIX form of a path, for path-based CSV matching."""
    text = clean_cell(value).replace("\\", "/").strip("/")
    return text.casefold()


def path_stem(value) -> str:
    text = normalize_path(value)
    if not text:
        return ""
    base = text.rsplit("/", 1)[-1]
    for extension in NETWORK_EXTENSIONS:
        if base.endswith(extension):
            return base[: -len(extension)]
    return os.path.splitext(base)[0]


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
