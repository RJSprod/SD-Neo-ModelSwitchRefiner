"""Using an artifact that is already on the machine instead of downloading it.

The model is one 16-27 GiB file, and a connection that cannot carry it is not
something setup can fix. Anyone who already has that file — downloaded by hand,
copied from another install, pulled with a download manager that resumes better
than we can — should be able to hand it over instead.

What arrives here is treated exactly like something downloaded: checked against
the pinned SHA-256 before it is installed, and put in the same folder under the
install root. The default is to *move* it rather than copy it, because the point
of supplying a 21 GiB file you already have is not to end up with two of them.

The folder, and not the file name. A supplied file keeps the name it arrived
with, because it is a file the person handing it over still has to recognise:
renaming ``MyMerge-Q5_K_M.gguf`` to the pinned build's name leaves a models
folder claiming to hold something it does not, and leaves them looking for a
file that no longer exists under any name they chose. Nothing downstream needs
a particular name — what was installed is recorded in the state file as the
path it actually went to, and that is the only thing that ever reads it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .manifest import Component
from .verifier import verify

Progress = Callable[[int, int], None]

COPY_BLOCK = 8 * 1024 * 1024


@dataclass(frozen=True)
class LocalSource:
    """A file on this machine to install instead of downloading a component."""

    path: Path
    move: bool = True      # a copy would leave the user with two 21 GiB files
    checked: bool = False  # the caller already hashed it and accepted the answer


class SourceMismatch(ValueError):
    """A supplied file is not the artifact the manifest pins."""


def human(size: int) -> str:
    """A size in the unit that makes the difference between two of them visible:
    a truncated download is "174.74 MiB", not "0.17 GiB"."""
    for unit, scale in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if size >= scale: return f"{size / scale:.2f} {unit}"
    return f"{size} bytes"


def size_problem(component: Component, path: Path) -> str | None:
    """Why ``path`` cannot be ``component``, from its size alone.

    Instant, and it catches the two ordinary mistakes — the wrong quantization
    and a half-downloaded file — before anything spends minutes reading 21 GiB.
    """
    if not path.is_file():
        return f"{path} is not a file"
    actual = path.stat().st_size
    if component.size is not None and actual != component.size:
        return (f"{path.name} is {human(actual)}, but {component.component_id} "
                f"is {human(component.size)}")
    return None


def inspect(component: Component, path: Path, progress: Progress | None = None) -> None:
    """Raise ``SourceMismatch`` unless ``path`` is byte-for-byte the pinned file."""
    problem = size_problem(component, path)
    if problem:
        raise SourceMismatch(problem)
    try:
        verify(path, component.size, component.sha256, progress)
    except ValueError as exc:
        raise SourceMismatch(f"{path.name} does not match the pinned SHA-256 for {component.component_id}") from exc


def installed_as(destination: Path, source: LocalSource) -> Path:
    """Where a supplied file lands: the manifest's folder, under its own name."""
    return destination.with_name(source.path.expanduser().resolve().name)


def adopt(component: Component, destination: Path, source: LocalSource,
          progress: Progress | None = None) -> Path:
    """Install ``source`` as ``component``. Returns the installed path.

    ``destination`` is where a download of this component would have gone. Only
    its folder is used — see the module docstring — so the path returned is not
    necessarily the one passed in, and callers must record what comes back
    rather than what they asked for.
    """
    path = source.path.expanduser().resolve()
    if not source.checked:
        inspect(component, path, progress)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An abandoned download of the same artifact is now dead weight, and a
    # part file at full size would block a later resume anyway. It is looked for
    # under the manifest's own name, which is what that download would have used.
    destination.with_name(destination.name + ".part").unlink(missing_ok=True)
    installed = installed_as(destination, source)
    if installed.is_file() and installed.samefile(path):
        if progress: progress(1, 1)
        return installed
    _install(path, installed, component.size or path.stat().st_size, move=source.move, progress=progress)
    return installed


def _install(source: Path, destination: Path, size: int, *, move: bool,
             progress: Progress | None) -> None:
    if move:
        try:
            os.replace(source, destination)   # same volume: a rename, and instant
            if progress: progress(size, size)
            return
        except OSError:
            pass                              # different volume, or a locked file
    free = shutil.disk_usage(destination.parent).free
    if free < size:
        raise OSError(f"{destination.parent} has {human(free)} free and the file needs "
                      f"{human(size)}. Choose an install directory with room.")
    partial = destination.with_name(destination.name + ".part")
    done = 0
    with source.open("rb") as reader, partial.open("wb") as writer:
        for block in iter(lambda: reader.read(COPY_BLOCK), b""):
            writer.write(block); done += len(block)
            if progress: progress(done, size or 1)
        writer.flush(); os.fsync(writer.fileno())
    os.replace(partial, destination)
    if move:
        # Only now, with the file safely at its destination: an interrupted copy
        # must never be able to leave the machine with neither.
        source.unlink(missing_ok=True)
