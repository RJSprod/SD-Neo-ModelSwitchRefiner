from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

Progress = Callable[[int, int], None]


def digest_of(path: Path, progress: Progress | None = None) -> str:
    """SHA-256 of a file. The progress callback is not decoration: hashing a
    21 GiB model takes minutes, and a silent minute reads as a hang."""
    total = path.stat().st_size; digest = hashlib.sha256(); done = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block); done += len(block)
            if progress: progress(done, total or 1)
    return digest.hexdigest()


def verify(path: Path, size: int | None, sha256: str, progress: Progress | None = None) -> None:
    if size is not None and path.stat().st_size != size: raise ValueError(f"Size mismatch for {path.name}")
    if digest_of(path, progress).casefold() != sha256.casefold(): raise ValueError(f"SHA-256 mismatch for {path.name}")
