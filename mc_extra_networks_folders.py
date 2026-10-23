"""Per-UI-Preset folder assignments for the extra-network browsers.

Forge Neo's *UI Preset* quick setting (``shared.opts.forge_preset``) already
switches the checkpoint, the modules and the sampling defaults for an
architecture. This module adds the missing half: which **folders** of each
extra-network type belong to a preset.

An assignment is a folder plus one explicit flag - whether its subfolders come
with it - so picking ``Lora/styles`` can mean *just this folder* or *this whole
subtree*. Assignments live in a small JSON file in the LoRA data root
(``mc_lora_paths``) so they survive a settings reset, and they are keyed by
preset and by extra-network page (``lora``, ``checkpoints``,
``textual_inversion``, ...).

Nothing here imports Gradio or ``modules``; the filter
(``mc_extra_networks_filter``) and the editor (``mc_extra_networks_folders_ui``)
build on it.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import mc_lora_paths
from mc_lora_common import log, normalize_key, opt, report

CONFIG_VERSION = 1
CONFIG_NAME = mc_lora_paths.FOLDERS_FILENAME

# Used when ``modules_forge.presets`` cannot be imported (tests, or a Forge build
# that moved it). The real list always wins when it is available.
FALLBACK_PRESETS = (
    "sd",
    "xl",
    "flux",
    "klein",
    "qwen",
    "lumina",
    "zit",
    "wan",
    "anima",
    "ernie",
    "pid",
    "krea",
    "qwen21",
    "moss",
)


@dataclass(frozen=True)
class Rule:
    """One assigned folder. ``subfolders`` is the user's explicit choice."""

    path: str
    subfolders: bool = False

    def as_dict(self) -> dict:
        return {"path": self.path, "subfolders": bool(self.subfolders)}


# --------------------------------------------------------------------- paths


def normalize_dir(value) -> str:
    """Absolute, comparable form of a directory path (case-folded on Windows)."""
    text = str(value or "").strip()
    if not text:
        return ""
    absolute = os.path.abspath(os.path.expanduser(text))
    normalized = os.path.normcase(absolute)
    # ``normpath`` already collapsed separators; only a trailing one can remain,
    # and only for a root such as ``C:\`` or ``/``, which must keep it.
    if len(normalized) > 1 and normalized.endswith(os.sep):
        stripped = normalized.rstrip(os.sep)
        normalized = stripped or normalized
    return normalized


def store_dir(value) -> str:
    """The form written to the JSON file: absolute, but not case-folded."""
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.abspath(os.path.expanduser(text))


def is_within(parent: str, child: str) -> bool:
    """True when ``child`` sits below ``parent`` (both already normalized)."""
    if not parent or not child:
        return False
    if parent == child:
        return False
    prefix = parent if parent.endswith(os.sep) else parent + os.sep
    return child.startswith(prefix)


def matches(rules, directory) -> bool:
    """Is ``directory`` covered by any of ``rules``?"""
    target = normalize_dir(directory)
    if not target:
        return False
    for rule in rules or ():
        base = normalize_dir(getattr(rule, "path", ""))
        if not base:
            continue
        if target == base:
            return True
        if getattr(rule, "subfolders", False) and is_within(base, target):
            return True
    return False


def file_allowed(filename, rules) -> bool:
    """Is the file at ``filename`` inside an assigned folder?"""
    text = str(filename or "").strip()
    if not text:
        # Nothing to place; keep it rather than hide a network we cannot judge.
        return True
    return matches(rules, os.path.dirname(os.path.abspath(os.path.expanduser(text))))


# ------------------------------------------------------------------- presets


def preset_names() -> list[str]:
    """Every UI Preset Forge offers, with the active one guaranteed present."""
    names: list[str] = []
    try:
        from modules_forge.presets import PresetArch  # type: ignore

        names = [str(name) for name in PresetArch.choices()]
    except Exception:
        names = []

    if not names:
        names = list(FALLBACK_PRESETS)

    active = current_preset()
    if active and active not in names:
        names.append(active)
    return names


def current_preset() -> str:
    """The UI Preset Forge is on right now."""
    return str(opt("forge_preset", "") or "").strip()


def page_key(page) -> str:
    """Stable key for an extra-networks page (``lora``, ``checkpoints``, ...)."""
    for attribute in ("extra_networks_tabname", "name", "title"):
        value = getattr(page, attribute, "") if not isinstance(page, str) else page
        if value:
            return normalize_key(value)
    return ""


# --------------------------------------------------------------------- store


class PresetFolderStore:
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
        return {"version": CONFIG_VERSION, "presets": {}}

    def load(self, *, force: bool = False) -> dict:
        with self._lock:
            if self._data is not None and not force:
                return self._data

            data = self._blank()
            if self._path.is_file():
                try:
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    log(f"the folder assignments file could not be read ({type(exc).__name__}); starting from an empty assignment list")
                    raw = None
                if isinstance(raw, dict):
                    presets = raw.get("presets")
                    if isinstance(presets, dict):
                        data["presets"] = {
                            str(preset): self._clean_pages(pages)
                            for preset, pages in presets.items()
                            if isinstance(pages, dict)
                        }

            self._data = data
            return self._data

    @staticmethod
    def _clean_pages(pages: dict) -> dict:
        cleaned: dict[str, list[dict]] = {}
        for key, entries in pages.items():
            if not isinstance(entries, (list, tuple)):
                continue
            rules: list[dict] = []
            seen: set[str] = set()
            for entry in entries:
                if isinstance(entry, str):
                    entry = {"path": entry, "subfolders": False}
                if not isinstance(entry, dict):
                    continue
                path = store_dir(entry.get("path", ""))
                if not path:
                    continue
                marker = normalize_dir(path)
                if marker in seen:
                    continue
                seen.add(marker)
                rules.append({"path": path, "subfolders": bool(entry.get("subfolders", False))})
            if rules:
                cleaned[normalize_key(key)] = rules
        return cleaned

    def save(self) -> bool:
        with self._lock:
            data = self.load()
            payload = json.dumps(data, indent=2, ensure_ascii=False)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, self._path)
            except OSError:
                report("could not write the folder assignments file")
                try:
                    temporary.unlink()
                except OSError:
                    pass
                return False
            return True

    def reload(self) -> dict:
        return self.load(force=True)

    # -- queries ----------------------------------------------------------

    def rules(self, preset: str, page) -> list[Rule]:
        key = page_key(page)
        preset = str(preset or "").strip()
        if not preset or not key:
            return []
        with self._lock:
            entries = self.load()["presets"].get(preset, {}).get(key, [])
            return [Rule(entry["path"], bool(entry.get("subfolders", False))) for entry in entries]

    def all_rules(self, preset: str) -> dict[str, list[Rule]]:
        preset = str(preset or "").strip()
        if not preset:
            return {}
        with self._lock:
            pages = self.load()["presets"].get(preset, {})
            return {
                key: [Rule(entry["path"], bool(entry.get("subfolders", False))) for entry in entries]
                for key, entries in pages.items()
            }

    def presets_in_use(self) -> list[str]:
        with self._lock:
            return sorted(preset for preset, pages in self.load()["presets"].items() if pages)

    # -- edits ------------------------------------------------------------

    def set_rules(self, preset: str, page, rules) -> bool:
        key = page_key(page)
        preset = str(preset or "").strip()
        if not preset or not key:
            return False

        entries: list[dict] = []
        seen: set[str] = set()
        for rule in rules or ():
            path = store_dir(getattr(rule, "path", rule))
            if not path:
                continue
            marker = normalize_dir(path)
            if marker in seen:
                continue
            seen.add(marker)
            entries.append({"path": path, "subfolders": bool(getattr(rule, "subfolders", False))})

        with self._lock:
            data = self.load()
            pages = data["presets"].setdefault(preset, {})
            if entries:
                pages[key] = entries
            else:
                pages.pop(key, None)
            if not pages:
                data["presets"].pop(preset, None)
            return self.save()

    def clear(self, preset: str, page=None) -> bool:
        with self._lock:
            data = self.load()
            preset = str(preset or "").strip()
            if page is None:
                data["presets"].pop(preset, None)
            else:
                pages = data["presets"].get(preset, {})
                pages.pop(page_key(page), None)
                if not pages:
                    data["presets"].pop(preset, None)
            return self.save()


_STORE: PresetFolderStore | None = None
_STORE_LOCK = threading.Lock()


def config_path() -> Path:
    return mc_lora_paths.folders_config_path()


def get_store() -> PresetFolderStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = PresetFolderStore(config_path())
        return _STORE


# ----------------------------------------------------------- folder listing


def subfolders_of(root, *, include_hidden: bool = False, limit: int = 4000) -> list[str]:
    """``root`` and every directory below it, depth-first and naturally sorted."""
    base = str(root or "").strip()
    if not base:
        return []
    base = os.path.abspath(os.path.expanduser(base))
    if not os.path.isdir(base):
        return []

    found = [base]
    for current, directories, _files in os.walk(base, followlinks=True):
        directories.sort(key=str.casefold)
        keep = []
        for name in directories:
            if not include_hidden and name.startswith("."):
                continue
            keep.append(name)
            found.append(os.path.join(current, name))
            if len(found) >= limit:
                break
        directories[:] = keep
        if len(found) >= limit:
            log(f"a folder list was truncated at {limit} entries")
            break
    return found


def folder_choices(roots, *, include_hidden: bool = False) -> list[tuple[str, str]]:
    """``(label, absolute path)`` for every root and subfolder, deduplicated.

    The label mirrors what the network browser shows - the root's own name and
    the path below it - so a folder is recognisable without reading the whole
    absolute path.
    """
    choices: list[tuple[str, str]] = []
    seen: set[str] = set()
    for root in roots or ():
        base = str(root or "").strip()
        if not base:
            continue
        base = os.path.abspath(os.path.expanduser(base))
        label_root = os.path.basename(base.rstrip(os.sep)) or base
        for path in subfolders_of(base, include_hidden=include_hidden):
            marker = normalize_dir(path)
            if marker in seen:
                continue
            seen.add(marker)
            relative = os.path.relpath(path, base)
            if relative in (".", ""):
                label = f"{label_root}/"
            else:
                label = f"{label_root}/{relative.replace(os.sep, '/')}/"
            choices.append((label, path))
    return choices
