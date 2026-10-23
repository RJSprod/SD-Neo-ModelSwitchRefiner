"""SHA256 hashing of network files, with a persistent on-disk cache.

The CSV produced by the Civitai tool identifies a LoRA by the SHA256 of the whole
file, which is *not* the same value Forge stores as ``NetworkOnDisk.hash`` (that
one prefers the kohya ``sshs_model_hash`` from the safetensors header). This
module therefore computes and caches the full-file digest itself, and can also
produce the AddNet/kohya digest so CSVs written by other tools still match.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import mc_lora_paths
from mc_lora_common import log, report

CHUNK_SIZE = 1024 * 1024
CACHE_VERSION = 1


class HashCache:
    """``{abspath: {size, mtime, sha256, addnet}}`` persisted as JSON."""

    def __init__(self, filename: Path | None = None):
        self._path = Path(filename) if filename else None
        self._lock = threading.RLock()
        self._data: dict[str, dict] = {}
        self._dirty = False
        self._loaded = False

    @property
    def path(self) -> Path:
        """Under the LoRA data root, resolved when first needed: importing this
        module makes no folder, and a test points the cache at its own file."""
        if self._path is not None:
            return self._path
        return mc_lora_paths.cache_dir() / "hashes.json"

    # ------------------------------------------------------------------ io

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError:
                return
            except (OSError, ValueError) as exc:
                log(f"the hash cache could not be read ({type(exc).__name__}); starting empty")
                return

            if not isinstance(payload, dict):
                return
            if payload.get("version") != CACHE_VERSION:
                return
            entries = payload.get("entries")
            if isinstance(entries, dict):
                self._data = {k: v for k, v in entries.items() if isinstance(v, dict)}

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            snapshot = {"version": CACHE_VERSION, "entries": dict(self._data)}
            self._dirty = False

        temporary = self.path.with_suffix(".json.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle)
            os.replace(temporary, self.path)
        except OSError as exc:
            log(f"could not write the hash cache ({exc.strerror or type(exc).__name__})")
            try:
                temporary.unlink()
            except OSError:
                pass

    # --------------------------------------------------------------- lookup

    @staticmethod
    def _stat(path) -> tuple[int, int] | None:
        try:
            info = os.stat(path)
        except (OSError, TypeError, ValueError):
            return None
        return int(info.st_size), int(info.st_mtime)

    def peek(self, path, kind: str = "sha256") -> str | None:
        """Return a cached digest without ever reading the file body."""
        self.load()
        stamp = self._stat(path)
        if stamp is None:
            return None

        with self._lock:
            entry = self._data.get(os.path.abspath(str(path)))

        if not entry or entry.get("size") != stamp[0] or entry.get("mtime") != stamp[1]:
            return None
        value = entry.get(kind)
        return value or None

    def store(self, path, kind: str, value: str) -> None:
        stamp = self._stat(path)
        if stamp is None:
            return
        key = os.path.abspath(str(path))
        with self._lock:
            entry = self._data.get(key)
            if not entry or entry.get("size") != stamp[0] or entry.get("mtime") != stamp[1]:
                entry = {"size": stamp[0], "mtime": stamp[1]}
                self._data[key] = entry
            entry[kind] = value
            self._dirty = True

    def prune(self) -> int:
        """Drop entries whose file is gone. Returns the number removed."""
        self.load()
        with self._lock:
            stale = [key for key in self._data if not os.path.isfile(key)]
            for key in stale:
                del self._data[key]
            if stale:
                self._dirty = True
        self.save()
        return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._dirty = True
            self._loaded = True
        self.save()

    def __len__(self) -> int:
        self.load()
        with self._lock:
            return len(self._data)


CACHE = HashCache()


# ------------------------------------------------------------------ digests


def _digest(path: Path, *, addnet: bool, cancel=None, progress=None) -> str | None:
    """Stream the file into SHA256. ``addnet`` skips the safetensors header."""
    digest = hashlib.sha256()
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if addnet:
                header = handle.read(8)
                if len(header) != 8:
                    return None
                offset = int.from_bytes(header, "little") + 8
                if offset <= 0 or offset > size:
                    return None
                handle.seek(offset)
                size -= offset

            done = 0
            while True:
                if cancel is not None and cancel.is_set():
                    return None
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                done += len(chunk)
                if progress is not None and size > 0:
                    progress(done, size)
    except OSError as exc:
        log(f"could not hash a network file ({exc.strerror or type(exc).__name__})")
        return None

    return digest.hexdigest().lower()


def sha256_file(path, *, cache: HashCache | None = None, cancel=None, progress=None) -> str | None:
    """Full-file SHA256 - the digest the Civitai `by-hash` endpoint expects."""
    path = Path(path)
    store = CACHE if cache is None else cache

    cached = store.peek(path, "sha256")
    if cached:
        return cached

    value = _digest(path, addnet=False, cancel=cancel, progress=progress)
    if value:
        store.store(path, "sha256", value)
    return value


def addnet_sha256_file(path, *, cache: HashCache | None = None, cancel=None) -> str | None:
    """kohya-ss "AddNet" digest: SHA256 of the tensor payload only."""
    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".safetensor"):
        return None

    store = CACHE if cache is None else cache
    cached = store.peek(path, "addnet")
    if cached:
        return cached

    value = _digest(path, addnet=True, cancel=cancel)
    if value:
        store.store(path, "addnet", value)
    return value


def known_hash(path) -> str | None:
    """Cached SHA256 only - never reads the file. Used on hot UI paths."""
    return CACHE.peek(Path(path), "sha256")


def flush() -> None:
    try:
        CACHE.save()
    except Exception:
        report("failed to flush the hash cache")
