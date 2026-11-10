"""The language model's control vector: a push applied inside its layers.

The third of the three ways put to the user for "a word injection into the mind
of the LLM" -- "I like all three options, give it a menu for 'Injection'"
(2026-11-09). The other two are a thread's (:mod:`mc_llm_injection`). This one
is the model's.

**What it is.** A control vector is one direction per layer, added to the
model's hidden state at that layer for every token it reads and writes. Made
from pairs of prompts that differ in one thing -- happy against sad, terse
against wordy -- it pushes every reply towards one side of that difference
without anything being said in the prompt. llama.cpp takes one only when
llama-server starts (``--control-vector``), for every request on that server:
every conversation, and every prompt the model writes for the image side and
for Mini Paint NEO, while the model is that one. Choosing one, or changing its
strength or layers, starts the server again, and every conversation is read
again at its next message.

**Where vectors come from.** llama.cpp's ``llama-cvector-generator`` makes them
from a model and two files of prompts. It is not in any of llama.cpp's Windows
release archives: it is built only when the backends are linked in
(``tools/CMakeLists.txt``: ``NOT GGML_BACKEND_DL``), and every release links
them dynamically -- the b10621 archives this extension installs list
twenty-one programs and not that one. So a vector is made elsewhere, for the
same model, and put in this extension's folder (:func:`folder`) or added from
the menu (:func:`add`). Each is read (:func:`read`) and checked against the
model the language model runs (:func:`fits`): a vector is one model's, by its
width and its layers, and llama-server will not start with one that is not.

**What is passed.** Not the file itself: a copy written here with the strength
and the layers taken in (:func:`applied`), given with ``--control-vector``.
``--control-vector-scaled`` takes ``FILE:STRENGTH`` and splits it at the colon,
which a Windows path has one of already (``common/arg.cpp``: two parts or it
refuses the start). A vector llama-server will not start with all the same is
set aside after the failed start and the server started without it
(:func:`set_aside_after_failure`); the menu says so until another is chosen.

Kept per model (``control_vector.json``), by the model file's name, so a model
chosen again has its vector again. Never logged: a vector's file name is the
user's.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import struct
import sys
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("model_chain")

DIRNAME = "control_vectors"
APPLIED_DIRNAME = "applied"
FILE_NAME = "control_vector.json"
VERSION = 1
FLAG = "--control-vector"
SUFFIX = ".gguf"

DEFAULT_STRENGTH = 1.0
STRENGTH_LIMIT = 8.0
"""The furthest the strength goes either way. A vector from llama.cpp's
generator is one unit long at each layer; a negative strength pushes the other
way."""

FIRST_LAYER_FRACTION = 0.3
"""Where the layers a vector is applied to start by default, as a share of the
model's: llama.cpp's own advice is that a vector works better past the first
ten of thirty-two (``tools/cvector-generator/README.md``)."""

MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_WIDTH = 65536
MAX_TENSORS = 4096
F32 = 0
ALIGNMENT = 32
ARCHITECTURE = "controlvector"

RESTART_WAIT_SECONDS = 600
"""How long a change waits for the model to be idle before giving up on starting
it again at once; it then takes effect at the next start."""
RESTART_POLL_SECONDS = 2.0

_lock = threading.Lock()
_given: dict[str, str] = {}
"""What each model's last start was given, by model key: the applied copy's
digest, or ``""`` for none."""
_set_aside: set[str] = set()
"""Applied digests a start failed with, set aside until another is chosen."""
_waiting: dict[str, threading.Thread] = {}


class Refused(ValueError):
    """Something that cannot be done, in a sentence for the screen."""


@dataclass(frozen=True)
class Vector:
    """One vector file as read: its width and the layers it has a direction for."""

    name: str
    path: Path
    width: int = 0
    layers: tuple[int, ...] = ()
    hint: str = ""
    error: str = ""

    @property
    def valid(self) -> bool:
        return not self.error and self.width > 0 and bool(self.layers)


@dataclass(frozen=True)
class Choice:
    vector: str = ""
    strength: float = DEFAULT_STRENGTH
    first_layer: int = 0
    last_layer: int = 0

    def to_dict(self) -> dict:
        return {"vector": self.vector, "strength": self.strength,
                "first_layer": self.first_layer, "last_layer": self.last_layer}


# -- the folder ------------------------------------------------------------------ #


def folder() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / DIRNAME


def _applied_folder() -> Path:
    return folder() / APPLIED_DIRNAME


def _record_path() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / FILE_NAME


def vectors() -> list[Vector]:
    """Every vector file in the folder, read, by name."""
    where = folder()
    if not where.is_dir():
        return []
    found = []
    for path in sorted(where.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.casefold() == SUFFIX:
            found.append(read(path, header_only=True))
    return found


def _by_name(name: str) -> Vector | None:
    wanted = str(name or "").strip().casefold()
    for vector in vectors():
        if vector.name.casefold() == wanted:
            return vector
    return None


# -- reading and writing a vector ------------------------------------------------- #


def read(path, header_only: bool = False):
    """``path`` as a control vector: a :class:`Vector`, with ``error`` set for a
    file that is not one. With ``header_only`` off, also its directions, as
    ``(vector, {layer: array('f')})``."""
    import mc_gguf

    path = Path(path)
    name = path.name
    directions: dict[int, array] = {}

    def refused(reason: str):
        found = Vector(name=name, path=path, error=reason)
        return found if header_only else (found, {})

    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return refused("It is too large to be a control vector.")
        with path.open("rb") as handle:
            if handle.read(4) != mc_gguf.MAGIC:
                return refused("It is not a GGUF file.")
            version, = struct.unpack("<I", mc_gguf._exactly(handle, 4))
            if version < 2:
                return refused("It is a GGUF version this cannot read.")
            tensors, kvs = struct.unpack("<QQ", mc_gguf._exactly(handle, 16))
            if tensors > MAX_TENSORS:
                return refused("It has too many tensors to be a control vector.")
            reader = mc_gguf._Reader(handle, size)
            metadata = {}
            for _ in range(min(int(kvs), 100_000)):
                key = reader.string()
                kind, = struct.unpack("<I", reader.take(4))
                metadata[key] = reader.value(kind)
            infos = []
            for _ in range(int(tensors)):
                tensor = reader.string()
                dims, = struct.unpack("<I", reader.take(4))
                if dims > 4:
                    return refused("Its directions are not single rows of 32-bit numbers.")
                shape = struct.unpack(f"<{dims}Q", reader.take(8 * dims)) if dims else ()
                kind, offset = struct.unpack("<IQ", reader.take(12))
                infos.append((tensor, tuple(int(item) for item in shape), int(kind),
                              int(offset)))
            alignment = metadata.get("general.alignment", ALIGNMENT)
            alignment = int(alignment) if isinstance(alignment, int) and alignment > 0 \
                else ALIGNMENT
            start = handle.tell()
            data = start + (-start % alignment)
            width = 0
            for tensor, shape, kind, offset in infos:
                matched = re.fullmatch(r"direction\.(\d+)", tensor)
                if matched is None or int(matched.group(1)) < 1:
                    return refused("It has tensors other than a control vector's directions.")
                if kind != F32 or len(shape) != 1:
                    return refused("Its directions are not single rows of 32-bit numbers.")
                if not 0 < shape[0] <= MAX_WIDTH or (width and shape[0] != width):
                    return refused("Its directions are not all the same width.")
                width = shape[0]
                layer = int(matched.group(1))
                if header_only:
                    directions[layer] = array("f")
                    continue
                handle.seek(data + offset)
                values = array("f")
                values.frombytes(mc_gguf._exactly(handle, 4 * width))
                if sys.byteorder != "little":
                    values.byteswap()
                directions[layer] = values
    except (OSError, struct.error, mc_gguf.GgufError, ValueError, OverflowError):
        return refused("It could not be read as a GGUF file.")
    if not directions:
        return refused("It has no directions.")
    hint = metadata.get(f"{ARCHITECTURE}.model_hint")
    found = Vector(name=name, path=path, width=width, layers=tuple(sorted(directions)),
                   hint=hint if isinstance(hint, str) else "")
    return found if header_only else (found, directions)


def _kv_string(key: str, value: str) -> bytes:
    import mc_gguf

    encoded_key, encoded = key.encode("utf-8"), value.encode("utf-8")
    return (struct.pack("<Q", len(encoded_key)) + encoded_key
            + struct.pack("<I", mc_gguf.STRING)
            + struct.pack("<Q", len(encoded)) + encoded)


def _kv_int32(key: str, value: int) -> bytes:
    import mc_gguf

    encoded_key = key.encode("utf-8")
    return (struct.pack("<Q", len(encoded_key)) + encoded_key
            + struct.pack("<I", mc_gguf.INT32) + struct.pack("<i", int(value)))


def write(path, directions: dict, hint: str = "") -> None:
    """A control vector file of ``directions`` (layer -> numbers), as llama.cpp's
    generator writes one: ``direction.N`` rows of 32-bit numbers. Atomic."""
    path = Path(path)
    layers = sorted(directions)
    header = bytearray(b"GGUF" + struct.pack("<I", 3)
                       + struct.pack("<QQ", len(layers), 3))
    header += _kv_string("general.architecture", ARCHITECTURE)
    header += _kv_string(f"{ARCHITECTURE}.model_hint", hint or "")
    header += _kv_int32(f"{ARCHITECTURE}.layer_count", len(layers))
    body = bytearray()
    for layer in layers:
        values = array("f", directions[layer])
        name = f"direction.{layer}".encode("utf-8")
        header += struct.pack("<Q", len(name)) + name
        header += struct.pack("<I", 1) + struct.pack("<Q", len(values))
        header += struct.pack("<IQ", F32, len(body))
        if sys.byteorder != "little":
            values.byteswap()
        body += values.tobytes()
        body += b"\0" * (-len(body) % ALIGNMENT)
    header += b"\0" * (-len(header) % ALIGNMENT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(header)
        handle.write(body)
    os.replace(temporary, path)


# -- the model ------------------------------------------------------------------- #


def _model(configuration=None) -> Path | None:
    try:
        if configuration is None:
            import mc_llm_runtime

            configuration = mc_llm_runtime.config()
        model = getattr(configuration, "model", None)
        return Path(model) if model else None
    except Exception:
        logger.debug("Model Chain: could not read which language model is configured",
                     exc_info=True)
        return None


def _model_key(model) -> str:
    return Path(model).name.casefold() if model else ""


def _shape(model) -> tuple[int, int]:
    """``(width, layers)`` of ``model``; zeros when it cannot be read."""
    try:
        import mc_gguf

        described = mc_gguf.describe(model) if model else None
    except Exception:
        described = None
    if described is None:
        return 0, 0
    return int(described.embedding_length or 0), int(described.block_count or 0)


def fits(vector: Vector, model) -> str:
    """Why ``vector`` cannot be used with ``model``, or ``""`` when it can."""
    if not vector.valid:
        return vector.error or "It is not a control vector."
    width, layers = _shape(model)
    if not width or not layers:
        return "The language model's file could not be read to check it against."
    if vector.width != width:
        return f"It was made for a model {vector.width} wide; this one is {width}."
    if max(vector.layers) >= layers:
        return f"It has directions past this model's {layers} layers."
    return ""


def default_layers(layers: int) -> tuple[int, int]:
    """The layers a vector is applied to by default in a model of ``layers``."""
    if layers <= 1:
        return 1, 1
    first = max(1, round(layers * FIRST_LAYER_FRACTION))
    return min(first, layers - 1), layers - 1


def resolved(choice: Choice, layers: int) -> Choice:
    """``choice`` with its layers in the model's range, the defaults for none."""
    first, last = default_layers(layers)
    if choice.first_layer:
        first = choice.first_layer
    if choice.last_layer:
        last = choice.last_layer
    top = max(layers - 1, 1)
    first, last = min(max(first, 1), top), min(max(last, 1), top)
    if first > last:
        first, last = last, first
    return Choice(vector=choice.vector, strength=choice.strength, first_layer=first,
                  last_layer=last)


# -- what is chosen, per model ------------------------------------------------------- #


def _read_choices() -> dict:
    where = _record_path()
    if not where.is_file():
        return {}
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        logger.warning("Model Chain: the chosen control vectors could not be read; the "
                       "language model starts without one until one is chosen again")
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return {}
    found = {}
    for key, raw in models.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            continue
        vector = raw.get("vector")
        strength = raw.get("strength")
        first, last = raw.get("first_layer"), raw.get("last_layer")
        found[key] = Choice(
            vector=vector if isinstance(vector, str) else "",
            strength=_strength(strength) if isinstance(strength, (int, float))
            and not isinstance(strength, bool) else DEFAULT_STRENGTH,
            first_layer=first if isinstance(first, int) and not isinstance(first, bool)
            and first > 0 else 0,
            last_layer=last if isinstance(last, int) and not isinstance(last, bool)
            and last > 0 else 0)
    return found


def _write_choices(choices: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(_record_path(), {"version": VERSION,
                                       "models": {key: choice.to_dict()
                                                  for key, choice in choices.items()}})


def _strength(value) -> float:
    number = float(value)
    if number != number:
        return DEFAULT_STRENGTH
    return round(min(max(number, -STRENGTH_LIMIT), STRENGTH_LIMIT), 3)


def chosen(model=None) -> Choice:
    """What is chosen for ``model`` (the configured one by default)."""
    model = model if model is not None else _model()
    with _lock:
        return _read_choices().get(_model_key(model), Choice())


def choose(vector=None, *, strength=None, first_layer=None, last_layer=None) -> dict:
    """Choose the configured model's vector -- a file's name, or ``""`` for none --
    and its strength and layers; what is not given stays as it was. The server is
    started again with it as soon as nothing is being written. Answers
    :func:`view`. Raises :class:`Refused`."""
    model = _model()
    if model is None:
        raise Refused("Set up a language model first: a control vector is made for one model.")
    key = _model_key(model)
    with _lock:
        choices = _read_choices()
        before = choices.get(key, Choice())
    name = before.vector
    if vector is not None:
        name = str(vector).strip()
        if name:
            found = _by_name(name)
            if found is None:
                raise Refused("That vector is not in the folder any more.")
            why = fits(found, model)
            if why:
                raise Refused(why)
            name = found.name
    made = Choice(vector=name,
                  strength=before.strength if strength is None else _strength(strength),
                  first_layer=before.first_layer if first_layer is None
                  else max(int(first_layer), 0),
                  last_layer=before.last_layer if last_layer is None
                  else max(int(last_layer), 0))
    with _lock:
        choices = _read_choices()
        if made == Choice():
            choices.pop(key, None)
        else:
            choices[key] = made
        _write_choices(choices)
        _set_aside.clear()
    if made != before:
        _restart_when_idle(model)
    return view()


# -- what a start is given ------------------------------------------------------------ #


def _digest(vector: Vector, choice: Choice) -> str:
    stat = vector.path.stat()
    parts = [vector.name.casefold(), str(stat.st_size), str(int(stat.st_mtime)),
             f"{choice.strength:g}", str(choice.first_layer), str(choice.last_layer)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _planned(model, choice: Choice) -> tuple[Vector, Choice, str]:
    """The vector ``choice`` names, the choice with its layers in range, and the
    applied copy's digest. Writes nothing. Raises :class:`Refused`."""
    vector = _by_name(choice.vector)
    if vector is None:
        raise Refused("The chosen vector is not in the folder any more.")
    why = fits(vector, model)
    if why:
        raise Refused(why)
    _, layers = _shape(model)
    choice = resolved(choice, layers)
    if not any(choice.first_layer <= layer <= choice.last_layer for layer in vector.layers):
        raise Refused("It has no directions in the layers chosen.")
    return vector, choice, _digest(vector, choice)


def applied(model, choice: Choice | None = None) -> Path | None:
    """The file a start of ``model`` is given: the chosen vector with its
    strength and layers taken in, written once and kept while it is chosen.
    None for no vector. Raises :class:`Refused` for one that cannot be used."""
    choice = choice if choice is not None else chosen(model)
    if not choice.vector:
        return None
    vector, choice, digest = _planned(model, choice)
    target = _applied_folder() / f"{digest}{SUFFIX}"
    if not target.is_file():
        found, directions = read(vector.path)
        if not found.valid:
            raise Refused(found.error or "It is not a control vector.")
        taken = {layer: array("f", (value * choice.strength for value in values))
                 for layer, values in directions.items()
                 if choice.first_layer <= layer <= choice.last_layer}
        if not taken:
            raise Refused("It has no directions in the layers chosen.")
        write(target, taken, found.hint)
    _tidy(keep=target)
    return target


def _tidy(keep: Path) -> None:
    """Remove applied copies no longer chosen for any model."""
    try:
        wanted = {keep.name}
        for path in _applied_folder().glob(f"*{SUFFIX}"):
            if path.name not in wanted and not _in_use(path.name):
                path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Model Chain: could not tidy the applied control vectors", exc_info=True)


def _in_use(name: str) -> bool:
    with _lock:
        return any(name.startswith(digest) for digest in _given.values() if digest)


def launch_flags(configuration) -> list[str]:
    """``--control-vector <applied copy>`` for a start of ``configuration``'s model
    with a vector chosen for it, when the build has the flag; ``[]`` otherwise.
    A vector that cannot be used is left off with a line in the log."""
    model = _model(configuration)
    key = _model_key(model)
    choice = chosen(model) if model is not None else Choice()
    with _lock:
        _given[key] = ""
    if not choice.vector:
        return []
    import mc_llm_runtime

    if not mc_llm_runtime.runtime_supports(FLAG, configuration):
        logger.warning("Model Chain: this llama-server build does not take a control vector; "
                       "the language model starts without the one chosen")
        return []
    try:
        target = applied(model, choice)
    except Refused as refusal:
        logger.warning("Model Chain: the chosen control vector was left off this start: %s",
                       refusal)
        return []
    except OSError as exc:
        logger.warning("Model Chain: the chosen control vector could not be written for this "
                       "start (%s); the language model starts without it", type(exc).__name__)
        return []
    if target is None:
        return []
    digest = target.stem
    if digest in _set_aside:
        return []
    if "," in str(target):
        logger.warning("Model Chain: the language model's data folder has a comma in its "
                       "path, which llama-server cannot take in a control vector's; it "
                       "starts without one")
        return []
    with _lock:
        _given[key] = digest
    return [FLAG, str(target)]


def set_aside_after_failure() -> bool:
    """A start given a control vector failed: set the vector aside and say whether
    the start should be tried again without it. Once per choice."""
    with _lock:
        given = [(key, digest) for key, digest in _given.items() if digest]
        fresh = [(key, digest) for key, digest in given if digest not in _set_aside]
        if not fresh:
            return False
        for key, digest in fresh:
            _set_aside.add(digest)
            _given[key] = ""
    logger.warning("Model Chain: llama-server did not start with the chosen control vector; "
                   "it has been set aside and the language model started without it")
    return True


# -- starting the model again ------------------------------------------------------- #


def _servers_of(model):
    """The running servers that were started for ``model``."""
    try:
        import mc_llm_runtime

        found = []
        for runtime in mc_llm_runtime.registry.running():
            configuration = runtime.configuration()
            if _model_key(getattr(configuration, "model", None)) == _model_key(model):
                found.append(runtime)
        return found
    except Exception:
        logger.debug("Model Chain: could not list the running language models", exc_info=True)
        return []


def _busy() -> bool:
    try:
        import mc_broker

        return bool(mc_broker.llm_busy())
    except Exception:
        return True


def _stop(model) -> int:
    stopped = 0
    for runtime in _servers_of(model):
        try:
            runtime.stop()
            stopped += 1
        except Exception:
            logger.warning("Model Chain: could not stop the language model to change its "
                           "control vector", exc_info=True)
    return stopped


def _restart_when_idle(model) -> str:
    """Stop the servers of ``model`` so its next start takes the new choice: now
    when nothing is being written, else as soon as nothing is. ``"stopped"``,
    ``"waiting"`` or ``"idle"`` (nothing was running)."""
    if not _servers_of(model):
        return "idle"
    if not _busy():
        _stop(model)
        logger.info("Model Chain: the language model was stopped to take its new control "
                    "vector; it starts again at the next request")
        return "stopped"
    key = _model_key(model)
    with _lock:
        waiter = _waiting.get(key)
        if waiter is not None and waiter.is_alive():
            return "waiting"

        def wait():
            deadline = time.monotonic() + RESTART_WAIT_SECONDS
            while time.monotonic() < deadline:
                if not _servers_of(model):
                    return
                if not _busy():
                    _stop(model)
                    logger.info("Model Chain: the language model was stopped to take its new "
                                "control vector once it was idle")
                    return
                time.sleep(RESTART_POLL_SECONDS)
            logger.info("Model Chain: the language model stayed busy; its new control vector "
                        "takes effect at its next start")

        waiter = threading.Thread(target=wait, name="mc-control-vector-restart", daemon=True)
        _waiting[key] = waiter
    waiter.start()
    return "waiting"


# -- adding a vector -------------------------------------------------------------- #


_UNSAFE = re.compile(r"[^A-Za-z0-9 ._()\-]+")


def add(source, name: str = "") -> str:
    """Copy ``source`` into the folder as a vector, under a name made safe from
    ``name`` (or its own), never over another file. Returns the name it is kept
    under. Raises :class:`Refused` for a file that is not a control vector."""
    source = Path(source)
    found = read(source, header_only=True)
    if not found.valid:
        raise Refused(found.error or "That is not a control vector.")
    stem = _UNSAFE.sub("_", Path(name or source.name).stem).strip(" ._") or "vector"
    stem = stem[:80]
    where = folder()
    where.mkdir(parents=True, exist_ok=True)
    target = where / f"{stem}{SUFFIX}"
    number = 2
    while target.exists():
        target = where / f"{stem}-{number}{SUFFIX}"
        number += 1
    shutil.copyfile(source, target)
    return target.name


# -- what the menus show ------------------------------------------------------------ #


def view() -> dict:
    """The model's control vector as the menus show it: the model, the vectors in
    the folder with whether each fits it, what is chosen, and where that stands.
    No path: the folder is named in words."""
    model = _model()
    width, layers = _shape(model) if model is not None else (0, 0)
    choice = resolved(chosen(model), layers) if model is not None else Choice()
    listed = []
    for vector in vectors():
        why = fits(vector, model) if model is not None else "No language model is set up."
        listed.append({"name": vector.name, "fits": not why, "why": why,
                       "layers": len(vector.layers), "hint": vector.hint})
    key = _model_key(model)
    state, said = "off", "No control vector: the model runs as it is."
    if model is None:
        state, said = "unavailable", "Set up a language model first."
    elif choice.vector:
        running = bool(_servers_of(model))
        with _lock:
            given = _given.get(key, "")
            waiting = key in _waiting and _waiting[key].is_alive()
        try:
            digest = _planned(model, choice)[2]
        except (Refused, OSError) as problem:
            digest = ""
            state, said = "unusable", f"The chosen vector cannot be used: {problem}"
        if digest:
            if digest in _set_aside:
                state, said = ("set_aside", "llama-server would not start with this vector, "
                               "so the model runs without it. Choose again to try once more.")
            elif running and given == digest:
                state, said = "on", "On: every reply on this model is pushed by it."
            elif waiting:
                state, said = ("waiting", "The model starts again with it as soon as it has "
                               "finished writing.")
            else:
                state, said = "next_start", "Takes effect when the model next starts."
    return {"model": Path(model).name if model is not None else "", "width": width,
            "layers": layers, "vectors": listed, "vector": choice.vector,
            "strength": choice.strength, "first_layer": choice.first_layer,
            "last_layer": choice.last_layer, "state": state, "said": said,
            "folder": "the control_vectors folder in the language model's data folder"}


__all__ = ["APPLIED_DIRNAME", "Choice", "DEFAULT_STRENGTH", "DIRNAME", "FILE_NAME", "FLAG",
           "Refused", "STRENGTH_LIMIT", "Vector", "add", "applied", "choose", "chosen",
           "default_layers", "fits", "folder", "launch_flags", "read", "resolved",
           "set_aside_after_failure", "vectors", "view", "write"]
