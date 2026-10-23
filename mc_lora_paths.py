"""Where the LoRA details keep their files: one data root, like the LLM's and
the Voice Chat's.

``<WebUI data>/model_chain_lora/`` holds the CSV library (``library/``), the
downloaded gallery images (``images/``), the hash cache and the thumbnails
(``cache/``), the folder assignments per UI Preset (``preset_folders.json``)
and what the last Civitai fetch used (``last_used.json``). The library and
the image folder can each be pointed elsewhere by a setting. Without a host
the root lands beside this file, which is where the tests' stand-in host puts
every data root; ``tests/conftest.py`` fails a test that writes there.

The standalone ``SD-ForgeNeo-ExtendedLoraDetails`` extension kept the same
files at its own root; :func:`old_extension_root` finds that folder so
``mc_lora_settings`` can bring them over once.
"""

from __future__ import annotations

import os
from pathlib import Path

DIRNAME = "model_chain_lora"
LIBRARY_DIRNAME = "library"
IMAGES_DIRNAME = "images"
CACHE_DIRNAME = "cache"
FOLDERS_FILENAME = "preset_folders.json"
LAST_USED_FILENAME = "last_used.json"

OLD_EXTENSION_DIRNAME = "SD-ForgeNeo-ExtendedLoraDetails"
"""The standalone extension's folder under the WebUI's ``extensions``."""

OPT_LIBRARY_DIR = "model_chain_lora_library_dir"
OPT_IMAGE_DIR = "model_chain_lora_image_dir"


def extension_root() -> Path:
    """The extension directory, which is where this file is."""
    return Path(__file__).resolve().parent


def _webui_data_path() -> Path:
    try:
        from modules import paths

        base = Path(paths.data_path)
    except Exception:
        # No host: the extension root is the only directory we can be sure
        # exists. Tests and standalone imports land here.
        base = extension_root()
    return base


def _setting(name: str) -> str:
    try:
        from modules import shared

        return str(getattr(shared.opts, name, "") or "").strip()
    except Exception:
        return ""


def data_root() -> Path:
    """``<WebUI data>/model_chain_lora``. Creates nothing."""
    return (_webui_data_path() / DIRNAME).resolve()


def _configured(name: str, default: Path) -> Path:
    configured = _setting(name)
    return Path(configured).expanduser() if configured else default


def library_dir() -> Path:
    """The CSV library: the setting, else ``library/`` under the data root."""
    return _configured(OPT_LIBRARY_DIR, data_root() / LIBRARY_DIRNAME)


def image_dir() -> Path:
    """Downloaded gallery images: the setting, else ``images/`` under the root."""
    return _configured(OPT_IMAGE_DIR, data_root() / IMAGES_DIRNAME)


def cache_dir() -> Path:
    """The hash cache and the thumbnails. Always under the data root."""
    return data_root() / CACHE_DIRNAME


def folders_config_path() -> Path:
    return data_root() / FOLDERS_FILENAME


def last_used_path() -> Path:
    return data_root() / LAST_USED_FILENAME


def ensure(path: Path) -> Path:
    """``path``, made if it is missing. Raises when it cannot be made."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def extensions_dir() -> Path | None:
    """The WebUI's ``extensions`` folder, where the standalone extension lives."""
    try:
        from modules import paths_internal

        found = getattr(paths_internal, "extensions_dir", None)
        if found:
            return Path(found)
    except Exception:
        pass
    try:
        from modules import paths

        script_path = getattr(paths, "script_path", None)
        if script_path:
            return Path(script_path) / "extensions"
    except Exception:
        pass
    return None


def old_extension_root(extensions=None) -> Path | None:
    """The standalone extension's folder, when it is installed."""
    base = Path(extensions) if extensions is not None else extensions_dir()
    if base is None:
        return None
    try:
        for entry in os.scandir(base):
            if entry.is_dir() and entry.name.lower() == OLD_EXTENSION_DIRNAME.lower():
                return Path(entry.path)
    except OSError:
        return None
    return None
