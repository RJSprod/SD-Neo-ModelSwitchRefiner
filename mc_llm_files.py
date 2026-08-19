"""What a user typed, as a path -- and what is in a folder.

Two jobs, both of them about the same gap: the panel asks for paths in text
boxes, and a text box is a lossy way to name a file. This module is what turns
what somebody actually pasted into the file they actually meant, and what lets
:mod:`mc_llm_browse` offer a picker instead of asking them to paste at all.

Why cleaning is not paranoia
----------------------------
Every one of the transformations in :func:`clean` is something a real paste
carries in:

* **Quotes.** Windows Explorer's *Copy as path* -- the obvious way to get a
  path into a clipboard, and the one the panel's own placeholder invites --
  copies ``"C:\\models\\thing.gguf"`` *with the double quotes*. Pasted as-is
  there is no such file, and the resulting "there is nothing at ..." names a
  path that looks exactly right, which is about the least useful error message
  it is possible to produce.
* **A ``file://`` URL.** What a file dragged from a file manager onto a browser
  text box becomes, percent-escaping and all.
* **Environment variables.** ``%USERPROFILE%\\models`` and ``$HOME/models`` are
  how a path gets written down in a note or a forum post.
* **A wrapped line.** A path copied out of a chat window or a PDF arrives with
  the line break still in it.

None of these are ambiguous, and none of them is the user making a mistake.

Why a folder is an answer and not an error
------------------------------------------
"My models path" is, to most people, the folder the models are in. Asked for a
model and given that folder, the honest answers are "here is the one model in
it" when there is one, and "it holds these six -- which one?" when there are
several. Refusing both with "there is no model file at ..." is the behaviour
that sends somebody back to the file manager to do work this can do.

Sharded models get the same treatment for the same reason. llama.cpp is handed
``-00001-of-00003.gguf`` and finds the other two itself; handed the third it
loads a third of a model and fails oddly. Somebody who picked the wrong shard
out of a folder listing made a reasonable mistake, so it is corrected and
reported rather than passed through.

Nothing here imports the vendored package or Gradio. It is used by the setup
module, by the panel, and by the picker, and it has to be importable on an
installation where neither of those will import.
"""

from __future__ import annotations

import logging
import os
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

MODEL_SUFFIX = ".gguf"
"""What a llama.cpp model file is called.

Restated rather than imported from ``prompt_master.inference.model_choice`` so
this module answers on an installation where the vendored package will not
import -- which is the installation most likely to be typing a path into the
setup panel. ``tests/test_llm_files.py`` asserts the two have not drifted.
"""

PROJECTOR_HINTS = ("mmproj", "projector")
"""What a vision projector is usually called. Same source, same reason."""

SHARD = re.compile(r"^(?P<stem>.+?)-(?P<part>\d{5})-of-(?P<total>\d{5})$")
"""llama.cpp's split-model naming, as ``gguf-split`` writes it."""

_QUOTES = ('"', "'", "\u201c\u201d", "\u2018\u2019", "`")
"""Pairs a path arrives wrapped in. Two-character entries are open/close.

Backticks are here because a path copied out of anything that renders Markdown
comes wrapped in them. Angle brackets are deliberately *not*: they wrap a path
about as often, and unlike the rest of these they are legal in a filename on
every filesystem this runs on."""

_INVISIBLE = "\u200b\u200e\u200f\u202a\u202c\ufeff"
"""Zero-width and direction marks, which a copy out of a rendered document
carries and a filesystem does not."""

MAX_ENTRIES = 4000
"""How many entries of one folder the picker will list.

A models folder holds tens of files and a system folder holds thousands. The
limit is not a judgement about either -- it is what keeps a dropdown from being
handed a folder that would make the page unusable.
"""


class PathError(ValueError):
    """Something a user can act on, phrased as a sentence for the panel."""


@dataclass(frozen=True)
class Resolved:
    """A path, and anything worth saying about how it was arrived at."""

    path: Path
    notes: tuple[str, ...] = ()

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


# --------------------------------------------------------------------------- #
# What was typed
# --------------------------------------------------------------------------- #


def clean(text) -> str:
    """The path inside what somebody pasted. See the module docstring.

    Never raises and never guesses at a file: this only undoes the things a
    clipboard does to a path. Whether the result exists is a separate question,
    asked by the resolvers below.
    """
    value = str(text if text is not None else "")
    for character in _INVISIBLE:
        value = value.replace(character, "")
    value = value.replace("\u00a0", " ")

    # A path cannot contain a line break, so a pasted one means either a
    # wrapped line or two paths. The first non-empty line is the one that was
    # meant in the second case and the error message shows what was used in the
    # first, which is better than silently joining two paths into neither.
    for line in value.splitlines():
        if line.strip():
            value = line
            break
    else:
        value = value.strip()

    value = _unwrap(value.strip())
    value = _from_url(value)
    value = os.path.expandvars(value)
    value = os.path.expanduser(value)
    return _without_trailing_separator(value.strip())


def to_path(text) -> Path | None:
    """:func:`clean` as a ``Path``, or ``None`` when nothing was typed."""
    value = clean(text)
    return Path(value) if value else None


def _unwrap(value: str) -> str:
    """Strip quotes a copy wrapped the path in, however many layers deep."""
    changed = True
    while changed and len(value) >= 2:
        changed = False
        for pair in _QUOTES:
            opening, closing = (pair[0], pair[-1])
            if value.startswith(opening) and value.endswith(closing):
                value = value[1:-1].strip()
                changed = True
                break
    # One unmatched quote, which is what a half-selected copy produces.
    while value[:1] in {pair[0] for pair in _QUOTES}:
        value = value[1:].lstrip()
    while value[-1:] in {pair[-1] for pair in _QUOTES}:
        value = value[:-1].rstrip()
    return value


def _from_url(value: str) -> str:
    """``file:///C:/models/thing.gguf`` as a path, and anything else unchanged."""
    if not value.lower().startswith("file:"):
        return value
    parsed = urlparse(value)
    path = unquote(parsed.path)
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        # file://server/share/... is a UNC path, and only on Windows.
        return f"\\\\{parsed.netloc}{path}".replace("/", "\\")
    if re.match(r"^/[A-Za-z]:", path):
        # file:///C:/... -- the leading slash belongs to the URL, not the path.
        path = path[1:]
    return path


def _without_trailing_separator(value: str) -> str:
    """``C:\\models\\`` as ``C:\\models``, without eating a root."""
    trimmed = value.rstrip("/\\")
    if not trimmed or trimmed.endswith(":"):
        return value
    return trimmed


# --------------------------------------------------------------------------- #
# What was meant
# --------------------------------------------------------------------------- #


def resolve_model(text) -> Resolved:
    """The GGUF a user meant by ``text``. Raises :class:`PathError` otherwise."""
    path = to_path(text)
    if path is None:
        raise PathError("Enter the path to a .gguf file, or press Browse to find one.")

    notes: list[str] = []
    path = _existing(path, notes)

    if path.is_dir():
        return _model_in(path, notes)
    if path.suffix.casefold() != MODEL_SUFFIX:
        raise PathError(
            f"{path.name} is not a GGUF file. llama.cpp reads {MODEL_SUFFIX} models — point "
            f"this at one, or at the folder holding it."
        )
    return _first_shard(path, notes)


def resolve_projector(text, model: Path | None = None) -> Resolved | None:
    """The projector a user meant, or ``None`` when the box was left empty.

    Empty is a real answer -- a model with no projector is a perfectly good
    model that cannot be shown a picture -- so it is the one input here that is
    not an error.
    """
    path = to_path(text)
    if path is None:
        return None

    notes: list[str] = []
    path = _existing(path, notes)

    if path.is_dir():
        found = [entry for entry in _models_in(path) if looks_like_projector(entry)]
        if not found:
            raise PathError(f"{path} holds no file that looks like a vision projector. "
                            f"Leave the box empty to run the model text-only.")
        if len(found) > 1:
            raise PathError(f"{path} holds {len(found)} possible projectors "
                            f"({_listed(found)}). Point this at one of them.")
        path, notes = found[0], notes + [f"Chose {found[0].name} from that folder."]
    elif path.suffix.casefold() != MODEL_SUFFIX:
        raise PathError(f"{path.name} is not a GGUF file. A vision projector is an "
                        f"mmproj{MODEL_SUFFIX}.")

    if model is not None and path == Path(model):
        raise PathError("The vision projector cannot be the model file itself. Leave the box "
                        "empty for a text-only model.")
    if not looks_like_projector(path):
        # A warning rather than a refusal: nothing in a file name proves what a
        # file is, and publishers do not all use the same one.
        notes.append(f"{path.name} is not named like a projector — check it belongs to this "
                     f"model, or llama-server will refuse to start.")
    return Resolved(path, tuple(notes))


def resolve_runtime(text) -> Resolved:
    """The llama-server -- or the folder holding it -- that ``text`` names."""
    path = to_path(text)
    if path is None:
        raise PathError("Enter the path to llama-server, or to the folder holding it, "
                        "or press Browse to find it.")
    notes: list[str] = []
    return Resolved(_existing(path, notes), tuple(notes))


def looks_like_projector(path: Path | str) -> bool:
    name = Path(path).name.casefold()
    return any(hint in name for hint in PROJECTOR_HINTS)


def _existing(path: Path, notes: list[str]) -> Path:
    """``path`` if it is there, the file it differs from only in case if that
    is there, and a sentence naming what is wrong otherwise."""
    if path.exists():
        return path

    parent = path.parent
    if parent.is_dir():
        folded = path.name.casefold()
        for entry in _entries(parent):
            if entry.name.casefold() == folded:
                # Only a Linux host can produce this: the path is right and the
                # capitalisation is not, which is invisible to a user who
                # copied it off a Windows machine.
                notes.append(f"Matched {entry.name}, which differs only in capitalisation.")
                return entry
        near = _nearest(path.name, parent)
        raise PathError(
            f"There is nothing at {path}."
            + (f" {parent} holds {near} — did you mean that?" if near else
               f" {parent} holds no file by that name.")
        )

    missing = _deepest_missing(path)
    raise PathError(f"There is nothing at {path} — {missing} does not exist either. "
                    f"Check the drive and the folder, or press Browse.")


def _deepest_missing(path: Path) -> Path:
    """The first folder along ``path`` that is not there.

    Named rather than the leaf because a wrong drive letter and a typo in a
    folder name both present as "there is nothing at <full path>", and which
    one it is decides what to do about it.
    """
    missing = path.parent
    for parent in path.parents:
        if parent.is_dir():
            break
        missing = parent
    return missing


def _nearest(name: str, folder: Path) -> str | None:
    import difflib

    names = [entry.name for entry in _entries(folder)]
    found = difflib.get_close_matches(name, names, n=1, cutoff=0.7)
    return found[0] if found else None


def _model_in(folder: Path, notes: list[str]) -> Resolved:
    """The one model in ``folder``, or a sentence naming the choice to make."""
    found = [entry for entry in _models_in(folder) if not looks_like_projector(entry)]
    shards = _leading_shards(found)

    if len(shards) == 1:
        chosen = shards[0]
        notes.append(f"Chose {chosen.name}, the only model in that folder.")
        return _first_shard(chosen, notes)
    if len(shards) > 1:
        raise PathError(
            f"{folder} holds {len(shards)} models ({_listed(shards)}). Point this at the one "
            f"to run, or press Browse to pick it."
        )

    deeper = [entry for entry in _entries(folder)
              if entry.is_dir() and any(_models_in(entry))]
    if deeper:
        raise PathError(f"{folder} holds no {MODEL_SUFFIX} files, but {_listed(deeper)} "
                        f"{'do' if len(deeper) > 1 else 'does'}. Press Browse to open it.")
    raise PathError(f"{folder} holds no {MODEL_SUFFIX} files.")


def _first_shard(path: Path, notes: list[str]) -> Resolved:
    """Shard one of a split model, given any shard of it.

    llama.cpp is handed the first and finds the rest; handed the third it loads
    a third of a model. See the module docstring.
    """
    found = SHARD.match(path.stem)
    if found is None or found.group("part") == "00001":
        return Resolved(path, tuple(notes))
    first = path.with_name(f"{found.group('stem')}-00001-of-{found.group('total')}{path.suffix}")
    if not first.is_file():
        return Resolved(path, tuple(notes) + (
            f"{path.name} is part {int(found.group('part'))} of "
            f"{int(found.group('total'))} and the first part is not beside it. llama.cpp "
            f"needs every part in one folder.",))
    return Resolved(first, tuple(notes) + (
        f"Using {first.name}: it is part 1 of {int(found.group('total'))}, and llama.cpp "
        f"loads the rest itself.",))


def _models_in(folder: Path) -> list[Path]:
    return [entry for entry in _entries(folder)
            if entry.is_file() and entry.suffix.casefold() == MODEL_SUFFIX]


def _leading_shards(found: list[Path]) -> list[Path]:
    """``found`` with every shard but the first of each split model dropped.

    A folder holding one three-part model holds three files and one model, and
    saying "this folder holds 3 models" of it would be false.
    """
    kept = []
    for path in found:
        shard = SHARD.match(path.stem)
        if shard is None or shard.group("part") == "00001":
            kept.append(path)
    return kept


def _listed(paths, limit: int = 4) -> str:
    names = [Path(path).name for path in paths]
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


# --------------------------------------------------------------------------- #
# What is in a folder
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Listing:
    """One folder, as the picker draws it."""

    directory: Path
    folders: tuple[Path, ...] = ()
    files: tuple[Path, ...] = ()
    detail: str = ""
    """Why the listing is not what was asked for -- unreadable, truncated, or
    a folder that is not there and was resolved to the nearest one that is."""


def listing(directory, suffixes: tuple[str, ...] = (MODEL_SUFFIX,),
            hidden: bool = False) -> Listing:
    """What is in ``directory``, for the picker. Never raises.

    An unreadable folder, a folder that has been deleted and a folder with
    fifty thousand files in it are all things a picker is pointed at, and none
    of them is a reason for the panel to stop drawing.
    """
    wanted = Path(str(directory or "")).expanduser() if directory else Path.home()
    detail = ""
    resolved = wanted
    while not resolved.is_dir() and resolved.parent != resolved:
        resolved = resolved.parent
    if resolved != wanted:
        detail = f"{wanted} is not a folder; showing {resolved}."
    try:
        resolved = resolved.resolve()
    except OSError:
        pass

    entries = _entries(resolved)
    if len(entries) > MAX_ENTRIES:
        entries = entries[:MAX_ENTRIES]
        detail = " ".join(filter(None, (
            detail,
            f"{resolved} holds more than {MAX_ENTRIES:,} entries; showing the first "
            f"{MAX_ENTRIES:,}.")))

    folders, files = [], []
    for entry in entries:
        if not hidden and entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                folders.append(entry)
            elif not suffixes or entry.suffix.casefold() in suffixes:
                files.append(entry)
        except OSError:
            continue
    return Listing(resolved, tuple(folders), tuple(files), detail)


def parent_of(directory) -> Path:
    """One level up, and the same folder when there is no up left."""
    path = Path(str(directory or "")).expanduser()
    return path.parent if path.parent != path else path


def starting_folder(current, fallback: Path | None = None) -> Path:
    """Where a picker should open, given whatever is in the box it belongs to."""
    path = to_path(current)
    if path is not None:
        candidate = path if path.is_dir() else path.parent
        if candidate.is_dir():
            return candidate
    if fallback is not None and Path(fallback).is_dir():
        return Path(fallback)
    return next(iter(places()), Path.home())


def places() -> list[Path]:
    """Folders worth one click: this install's own, the host's, and the drives.

    Ordered by how likely each is to be where the models are, which is why the
    LLM data directory comes first and the drive roots come last.
    """
    import mc_llm_paths

    candidates: list[Path] = []

    def add(value) -> None:
        if not value:
            return
        try:
            path = Path(str(value)).expanduser()
        except (OSError, ValueError):
            return
        if path.is_dir() and path not in candidates:
            candidates.append(path)

    try:
        root = mc_llm_paths.data_root()
        add(root / "models")
        add(root)
    except Exception:
        logger.debug("Model Chain: could not read the LLM data directory", exc_info=True)

    add(os.environ.get(mc_llm_paths.ROOT_ENV))
    for name in ("models_path", "data_path", "script_path"):
        add(_host_path(name))
    add(Path.home() / "Downloads")
    add(Path.home())
    for drive in _drives():
        add(drive)
    return candidates


def _host_path(name: str):
    try:
        from modules import paths

        return getattr(paths, name, None)
    except Exception:
        return None


def _drives() -> list[Path]:
    if sys.platform != "win32":
        return [Path("/")]
    return [Path(f"{letter}:\\") for letter in string.ascii_uppercase
            if os.path.exists(f"{letter}:\\")]


def _entries(folder: Path) -> list[Path]:
    """``folder``'s contents, sorted for a human, and empty when unreadable."""
    try:
        found = list(os.scandir(folder))
    except OSError:
        return []
    found.sort(key=lambda entry: entry.name.casefold())
    return [Path(entry.path) for entry in found]
