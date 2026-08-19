"""Pointing an installed application at a different model.

Setup downloads a pinned model and the vision projector that belongs to it,
verifies both by SHA-256, and records where they went. That is the right
default and it is not the only thing anybody wants to run: a GGUF is a GGUF,
and somebody with a folder of them should be able to try one without
re-provisioning an install that is already complete.

So this is deliberately the small half of setup. It changes two lines of the
state file — which weights, and which projector — and nothing else. The runtime
stays where it is, the device stays what it was, and no download happens,
because none of those three has anything to do with which file the weights are
read out of. What it does not do is verify: a file the manifest does not pin
cannot be checked against a hash it has no entry for, and refusing everything
unpinned would refuse the whole feature.

Two things are worth stating about the projector, because they are what make
this more than a file picker:

*It is optional.* The pinned model ships beside its projector; an arbitrary
GGUF may have none, and one downloaded on its own usually does. A model with no
projector is a perfectly good model that cannot be shown a picture, so it is
installed as exactly that — ``mmproj`` recorded empty, ``--mmproj`` left off
the command line, and image requests refused with a sentence saying why rather
than by a server that will not start.

*It is chosen separately.* A projector has to match the model it was made for,
and nothing in a file name proves that it does. Pairing them by guesswork is
how a chat ends up describing a picture nobody attached, so the path is asked
for rather than inferred — with the file beside the model offered as the
suggestion it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from prompt_master.core.config import atomic_write_json, read_json
from prompt_master.core.paths import AppPaths

# What a llama.cpp model file is called. Everything llama.cpp has loaded since
# 2023 is GGUF; the older formats it dropped are not worth offering.
MODEL_SUFFIX = ".gguf"

# What a projector is usually called, in the two spellings publishers use.
PROJECTOR_HINTS = ("mmproj", "projector")

# The quantization inside a model's file name: Q4_K_M, IQ3_XXS, Q8_0, BF16 and
# the rest of the family. It is the one part of a name worth showing on a status
# line — "Model: Q6_K_P" is what the pinned install has always said there, and a
# hand-picked file should read the same way rather than as a 60-character stem.
_QUANTIZATION = re.compile(r"(?<![0-9A-Za-z])(I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32)(?![0-9A-Za-z])",
                           re.IGNORECASE)


@dataclass(frozen=True)
class Choice:
    """A model, and the projector that gives it eyes — or the absence of one."""

    model: Path
    mmproj: Path | None = None

    @property
    def sees(self) -> bool:
        return self.mmproj is not None


def recorded(paths: AppPaths) -> Choice | None:
    """What the state file names now, resolved, or ``None`` on a bare install.

    Best-effort by design: this fills in a dialog's boxes, and a state file that
    cannot be read is a reason to open that dialog empty rather than to refuse
    to open it.
    """
    try:
        state = read_json(paths.state_file)
    except (OSError, ValueError):
        return None
    model = _resolve(paths, state.get("model"))
    if model is None:
        return None
    return Choice(model, _resolve(paths, state.get("mmproj")))


def choose(paths: AppPaths, model: Path, mmproj: Path | None = None) -> dict:
    """Run ``model`` from now on, with ``mmproj`` if one is given.

    Returns the new state. The caller is expected to stop the server after
    this: llama-server holds the weights it was started with, so the state file
    saying otherwise changes nothing until it is restarted — which the next
    generation does on its own, because the signature it compares against no
    longer matches.
    """
    try:
        state = read_json(paths.state_file)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"This install's setup state cannot be read ({exc}). "
                           "Run Models and Hardware setup.") from exc
    if not state.get("runtime"):
        raise RuntimeError("This install has no llama.cpp runtime yet, so there is nothing to "
                           "run a model with. Run Models and Hardware setup first.")

    model = Path(model).expanduser()
    if not model.is_file():
        raise RuntimeError(f"There is no model file at {model}")
    if mmproj is not None:
        mmproj = Path(mmproj).expanduser()
        if not mmproj.is_file():
            raise RuntimeError(f"There is no vision projector at {mmproj}")

    updated = {**state,
               "model": paths.record(model),
               "mmproj": paths.record(mmproj) if mmproj is not None else "",
               # What the status line calls the model. The pinned installs put
               # their quantization here and this keeps doing that when the name
               # says one, so a status line reads the same either way.
               "quantization": describe(model)}
    atomic_write_json(paths.state_file, updated)
    return updated


def describe(model: Path) -> str:
    """The short name for a model file: its quantization, or its stem."""
    stem = Path(model).stem
    found = _QUANTIZATION.search(stem)
    return found.group(1).upper() if found else stem


def projector_beside(model: Path) -> Path | None:
    """A projector sitting next to ``model``, if one obviously is.

    A suggestion and never a decision — see the module docstring. It is offered
    because a model and its projector are two files from the same repository
    and are downloaded into the same folder nearly every time, and it is
    offered into a box the user can empty.
    """
    folder = Path(model).expanduser().parent
    if not folder.is_dir():
        return None
    candidates = [path for path in sorted(folder.glob(f"*{MODEL_SUFFIX}"))
                  if path != Path(model).expanduser()
                  and any(hint in path.name.casefold() for hint in PROJECTOR_HINTS)]
    return candidates[0] if candidates else None


def _resolve(paths: AppPaths, value) -> Path | None:
    if not value:
        return None
    try:
        return paths.locate(str(value))
    except ValueError:
        return None
