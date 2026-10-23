"""What the last Civitai fetch used, kept for the next one.

Two values are worth carrying across sessions: the API key that last worked, and
the folder that was last scanned. They seed the *Fetch from Civitai* tab, so a
new session starts where the previous one left off instead of on an empty key
field and whichever folder happens to sort first.

They live in a small JSON file in the LoRA data root (``mc_lora_paths``) —
next to ``preset_folders.json`` and for the same reason: this is the extension's
own state rather than a setting, so a ``config.json`` reset leaves it alone. The
file holds an API key, so it is written owner-only wherever file modes mean
anything.

Nothing here imports Gradio or ``modules``; the tab (``mc_lora_tab``) and the
job layer (``mc_lora_jobs``) build on it.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import mc_lora_paths
from mc_lora_common import log, opt, report

STATE_VERSION = 1
STATE_NAME = mc_lora_paths.LAST_USED_FILENAME

API_KEY = "civitai_api_key"
SCAN_FOLDER = "scan_folder"


class LastUsedStore:
    """The JSON file, read once and written atomically."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._data: dict | None = None

    @property
    def path(self) -> Path:
        return self._path

    # -- io ---------------------------------------------------------------

    @staticmethod
    def _blank() -> dict:
        return {"version": STATE_VERSION, API_KEY: "", SCAN_FOLDER: ""}

    def load(self, *, force: bool = False) -> dict:
        with self._lock:
            if self._data is not None and not force:
                return self._data

            data = self._blank()
            if self._path.is_file():
                try:
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    log(f"the last-used file could not be read ({type(exc).__name__}); nothing is remembered")
                    raw = None
                if isinstance(raw, dict):
                    for key in (API_KEY, SCAN_FOLDER):
                        value = raw.get(key)
                        if isinstance(value, str):
                            data[key] = value.strip()

            self._data = data
            return self._data

    def reload(self) -> dict:
        return self.load(force=True)

    def save(self) -> bool:
        with self._lock:
            payload = json.dumps(self.load(), indent=2, ensure_ascii=False)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(payload, encoding="utf-8")
                # The file carries an API key, so it stays readable only by the
                # account that wrote it on the platforms that can say so.
                try:
                    os.chmod(temporary, 0o600)
                except OSError:
                    pass
                os.replace(temporary, self._path)
            except OSError:
                report("could not write the last-used file")
                try:
                    temporary.unlink()
                except OSError:
                    pass
                return False
            return True

    # -- values -----------------------------------------------------------

    def get(self, key: str) -> str:
        with self._lock:
            return str(self.load().get(key, "") or "")

    def set(self, key: str, value) -> bool:
        text = str(value or "").strip()
        with self._lock:
            data = self.load()
            if data.get(key, "") == text:
                # Nothing changed - a fetch that reuses the same key or folder
                # costs no write at all.
                return True
            data[key] = text
            return self.save()


_STORE: LastUsedStore | None = None
_STORE_LOCK = threading.Lock()


def state_path() -> Path:
    return mc_lora_paths.last_used_path()


def get_store() -> LastUsedStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = LastUsedStore(state_path())
        return _STORE


# ------------------------------------------------------------------ api key


def remembering_api_key() -> bool:
    return bool(opt("model_chain_lora_remember_api_key", True))


def remembered_api_key() -> str:
    """The key the last accepted fetch used, or ``""`` when there is none."""
    if not remembering_api_key():
        return ""
    return get_store().get(API_KEY)


def remember_api_key(key) -> None:
    """Keep a key Civitai accepted. Never clears: only a better key replaces it."""
    text = str(key or "").strip()
    if not text or not remembering_api_key():
        return
    try:
        get_store().set(API_KEY, text)
    except Exception:  # noqa: BLE001 - a convenience must never break a fetch
        report("could not remember the Civitai API key")


def forget_api_key() -> None:
    """Drop the stored key — what turning the setting off does."""
    try:
        get_store().set(API_KEY, "")
    except Exception:  # noqa: BLE001
        report("could not forget the Civitai API key")


# ------------------------------------------------------------------- folder


def remembering_scan_folder() -> bool:
    return bool(opt("model_chain_lora_remember_scan_folder", True))


def remembered_scan_folder() -> str:
    """The folder the last fetch ran on, as long as it is still there."""
    if not remembering_scan_folder():
        return ""
    folder = get_store().get(SCAN_FOLDER)
    # A folder on a drive that is not mounted right now is not offered as the
    # default, but it is kept: it becomes the default again when it comes back.
    return folder if folder and os.path.isdir(folder) else ""


def remember_scan_folder(folder) -> None:
    text = str(folder or "").strip()
    if not text or not remembering_scan_folder():
        return
    absolute = os.path.abspath(os.path.expanduser(text))
    if not os.path.isdir(absolute):
        # A fetch aimed at something that is not a folder fails anyway; it must
        # not displace the folder that does work.
        return
    try:
        get_store().set(SCAN_FOLDER, absolute)
    except Exception:  # noqa: BLE001
        report("could not remember the scanned folder")
