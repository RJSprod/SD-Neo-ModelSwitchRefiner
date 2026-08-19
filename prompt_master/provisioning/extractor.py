from __future__ import annotations

import shutil, tempfile, zipfile
from pathlib import Path


def extract_zips_atomic(archives: list[Path], destination: Path) -> None:
    """Extract related archives into one directory without exposing a partial runtime."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for archive in archives:
            with zipfile.ZipFile(archive) as bundle:
                root = staging.resolve()
                for member in bundle.infolist():
                    target = (staging / member.filename).resolve()
                    if target != root and root not in target.parents: raise ValueError(f"Unsafe archive path: {member.filename}")
                bundle.extractall(staging)
        if destination.exists(): shutil.rmtree(destination)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise


def extract_zip_atomic(archive: Path, destination: Path) -> None:
    extract_zips_atomic([archive], destination)
