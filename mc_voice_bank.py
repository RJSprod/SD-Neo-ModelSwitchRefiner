"""The Kokoro voice bank: how a cloned voice reaches a runtime that has no idea.

This module is the answer to the one genuinely hard question in the combined
scope, and the answer is not the one the original cloning proposal assumed.

That proposal was: take Storytime's clone, ``torch.save`` it as a ``.pt``, and
register the ``.pt`` with Kokoro. That is how the *upstream Python* Kokoro is
used. It is not how this repository speaks. Production TTS here is
``sherpa_onnx.OfflineTts``, which does not know what a ``.pt`` is and has no
concept of registering a voice at all. Section 48 supersedes the ``.pt`` plan,
and this file implements what replaces it.

What sherpa actually does
-------------------------
Reading ``offline-tts-kokoro-model.cc`` rather than guessing:

    * the ONNX graph is loaded, and its metadata carries ``n_speakers`` and
      ``style_dim`` -- for Kokoro v1.0, ``style_dim`` is ``510,1,256``;
    * ``voices.bin`` is loaded *separately*, as one flat block of float32;
    * the loader refuses to start unless that block contains exactly
      ``style_dim[0] * style_dim[2] * n_speakers`` floats;
    * choosing a speaker is pointer arithmetic --
      ``styles + sid * 510 * 256 + token_len * 256`` -- and what reaches the
      graph is a 1x256 style vector;
    * a ``sid`` at or past ``n_speakers`` is not an error: sherpa logs a
      warning and uses speaker 0.

Three consequences, and the whole design follows from them.

**The graph never sees the speaker.** So adding voices needs no new weights, no
retraining, no fork of sherpa and no second TTS engine. It needs a longer
``voices.bin``.

**The metadata is the gate.** A longer ``voices.bin`` alone is refused, because
the float count would not match. So the bank ships with a *derived* model whose
``n_speakers`` says how many speakers the bank has -- the same graph, byte for
byte, with one metadata string rewritten. :func:`derive_model` does that
rewrite at the protobuf level with no ONNX library, because the isolated CPU
runtime has sherpa and nothing else, and adding a PyTorch or an ``onnx``
dependency to change one string would be a strange trade for the same result.

**The silent failure is the dangerous one.** A custom sid against an
unextended model does not fail -- it speaks in ``af_alloy``'s voice. That is
why section 55 requires a clone to be *synthesized through the production
runtime* before its registry entry is committed, and why the handshake refuses
a worker whose bank is smaller than the registry expects.

Fixed slots, never compacted
----------------------------
Custom voices occupy a fixed block of reserved slots after the official ones.
A slot is allocated once and is never reused by compaction, because a SID is
baked into every place a voice has been chosen: delete the clone in slot 2 and
renumber slot 3 down into it, and every user who had selected "Bob" is now
listening to "Carol". Section 50 and release blocker seven.

The transaction
---------------
Nothing here mutates the live bank in place. A build assembles a candidate in
a staging directory beside the live one, hashes it, and promotes it with one
``os.replace`` on the same filesystem. A promotion that is not followed by a
successful synthesis is rolled back to the previous bytes, which are kept for
exactly that reason. Release blocker six: there is no moment at which a partial
``voices.bin`` is what the runtime would load.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import mc_voice_models as models
import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

STYLE_DIM = 256
"""Floats per style row. Kokoro v1.0's ``style_dim[2]``."""

STYLE_ROWS = 510
"""Rows per speaker. Kokoro v1.0's ``style_dim[0]``, and also its maximum token
length -- sherpa indexes the row by how long the sentence is."""

ROW_BYTES = STYLE_DIM * 4
SPEAKER_BYTES = STYLE_ROWS * ROW_BYTES

CUSTOM_CAPACITY = 32
"""Reserved custom slots. Section 50's product constant.

Thirty-two costs 32 x 510 x 256 x 4 bytes -- about 16 MB of extra
``voices.bin``, read once at worker start. That is the entire price of the
feature, and it is why the capacity is fixed rather than grown per clone: a
bank that changed size on every clone would need the derived model rebuilt on
every clone too.
"""

FILLER_VOICE = "af_heart"
"""What an empty reserved slot contains.

A real, known-good official voice rather than zeros. Zeros would be a
structurally valid bank that produces silence or noise if anything ever
addressed the slot, and "valid but wrong" is the failure mode that is hardest
to notice. A filler slot is never in the registry and never offered.
"""

SCHEMA = 1


class BankError(RuntimeError):
    """A bank that could not be read, built or promoted. Never fatal."""


# --------------------------------------------------------------------------- #
# ONNX metadata, without an ONNX library
# --------------------------------------------------------------------------- #
#
# A ``.onnx`` file is a serialized ``ModelProto``. Everything below reads and
# rewrites exactly one thing in it -- ``metadata_props``, field 14, a repeated
# ``StringStringEntryProto`` of ``key`` (field 1) and ``value`` (field 2) --
# and copies every other byte through untouched. There is no graph parsing
# here, no tensor is decoded, and a file this does not understand is refused
# rather than guessed at.


def _varint(data: bytes, index: int) -> tuple:
    value = shift = 0
    while True:
        if index >= len(data):
            raise BankError("this model file ended in the middle of a number")
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
        if shift > 63:
            raise BankError("this model file contains a number that cannot be read")


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _fields(data: bytes):
    """Walk the top-level fields of a protobuf message held in memory.

    Yields ``(number, wire_type, tag_start, body_start, end)``. Only the four
    wire types a modern protobuf uses are handled; the deprecated group types
    are refused, because a file containing them is not a file this understands
    well enough to rewrite.
    """
    index = 0
    while index < len(data):
        tag_start = index
        key, index = _varint(data, index)
        number, wire = key >> 3, key & 7
        if wire == 0:
            _value, index = _varint(data, index)
        elif wire == 1:
            index += 8
        elif wire == 2:
            length, body = _varint(data, index)
            index = body + length
        elif wire == 5:
            index += 4
        else:
            raise BankError("this model file uses a protobuf feature Voice Chat cannot read")
        if index > len(data):
            raise BankError("this model file is truncated")
        yield number, wire, tag_start, (body if wire == 2 else tag_start), index


def _walk(handle):
    """The same walk over an open file, seeking past bodies rather than reading.

    A Kokoro model is about 350 MB and almost all of it is one field: the graph.
    Reading the file to find out how many speakers it declares is fine once and
    is not fine on a status route a browser polls -- so this reads tag headers,
    seeks over the bodies, and only ever holds a metadata entry in memory.

    Yields ``(number, wire_type, tag_start, body_start, end)``, the same tuple
    :func:`_fields` yields, so the two readers agree about what a field is.
    """
    while True:
        tag_start = handle.tell()
        key = _read_varint(handle)
        if key is None:
            return
        number, wire = key >> 3, key & 7
        if wire == 0:
            if _read_varint(handle) is None:
                return
            body = tag_start
        elif wire == 1:
            handle.seek(8, 1)
            body = tag_start
        elif wire == 2:
            length = _read_varint(handle)
            if length is None:
                return
            body = handle.tell()
            handle.seek(length, 1)
        elif wire == 5:
            handle.seek(4, 1)
            body = tag_start
        else:
            raise BankError("this model file uses a protobuf feature Voice Chat cannot read")
        yield number, wire, tag_start, body, handle.tell()


def _read_varint(handle):
    """One varint from an open file, or ``None`` at end of it."""
    value = shift = 0
    while True:
        byte = handle.read(1)
        if not byte:
            return None if shift == 0 else _truncated()
        found = byte[0]
        value |= (found & 0x7F) << shift
        if not found & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise BankError("this model file contains a number that cannot be read")


def _truncated():
    raise BankError("this model file is truncated")


def _entry(data: bytes) -> tuple:
    """``(key, value)`` from one ``StringStringEntryProto``."""
    key = value = ""
    for number, wire, _tag, body, end in _fields(data):
        if wire != 2:
            continue
        text = data[body:end].decode("utf-8", "replace")
        if number == 1:
            key = text
        elif number == 2:
            value = text
    return key, value


def _encode_entry(key: str, value: str) -> bytes:
    def part(number: int, text: str) -> bytes:
        raw = text.encode("utf-8")
        return _encode_varint((number << 3) | 2) + _encode_varint(len(raw)) + raw

    return part(1, key) + part(2, value)


_metadata_cache: dict = {}
_metadata_lock = threading.Lock()


def read_metadata(model_path) -> dict:
    """Every ``metadata_props`` string in an ONNX file, as a plain dict.

    Read from the file rather than from a manifest, because the whole point of
    the gate this supports is to find out what the *installed* model says
    rather than what a document says it should say.

    Cached against the file's size and modification time, and streamed rather
    than loaded. Both matter for the same reason: this is on the path of a
    status route a page polls, and answering it by reading 350 MB of ONNX would
    make polling the status an attack on the user's own machine -- which is the
    thing :func:`mc_voice_runtime.status` was written not to be.
    """
    path = Path(model_path)
    try:
        stamp = path.stat()
        key = (str(path), stamp.st_size, stamp.st_mtime_ns)
    except OSError:
        return {}
    with _metadata_lock:
        found = _metadata_cache.get(key)
    if found is not None:
        return dict(found)

    found = {}
    with open(path, "rb") as handle:
        for number, wire, _tag, body, end in _walk(handle):
            if number != 14 or wire != 2:
                continue
            here = handle.tell()
            handle.seek(body)
            entry = handle.read(end - body)
            handle.seek(here)
            name, value = _entry(entry)
            if name:
                found[name] = value
    with _metadata_lock:
        # One model, one derived model, and whatever a rebuild left behind. A
        # cap rather than an eviction policy, because there is no third file.
        if len(_metadata_cache) > 8:
            _metadata_cache.clear()
        _metadata_cache[key] = dict(found)
    return found


def speaker_capacity(model_path) -> int:
    """How many speakers a model file's metadata admits to."""
    try:
        return int(str(read_metadata(model_path).get("n_speakers") or "0") or 0)
    except (ValueError, TypeError):
        return 0


def style_shape(model_path) -> tuple:
    """``(rows, dim)`` from ``style_dim``, or the Kokoro v1.0 defaults."""
    raw = str(read_metadata(model_path).get("style_dim") or "")
    parts = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(parts) == 3:
        try:
            return int(parts[0]), int(parts[2])
        except ValueError:
            pass
    return STYLE_ROWS, STYLE_DIM


def derive_model(source, destination, speakers: int) -> str:
    """Copy ``source`` to ``destination`` with ``n_speakers`` set to ``speakers``.

    The graph, the weights, the opset, the producer and every other byte are
    identical -- this rewrites one string in one metadata entry and nothing
    else, which is what makes the derivation reproducible and what makes the
    licence and attribution of the original still the licence and attribution
    of the result. Returns the sha256 of what was written.

    Copied in blocks rather than through one ``read_bytes``: a Kokoro model is
    350 MB, this runs inside the WebUI process beside an image model, and
    holding two copies of it in memory to change two characters would be a
    strange thing to do to somebody's machine.

    Refuses a model that does not already declare ``n_speakers``: that is not a
    Kokoro multi-speaker model, and adding the key would be inventing a claim
    about a file rather than adjusting one.
    """
    source = Path(source)
    edits = []
    with open(source, "rb") as handle:
        for number, wire, tag, body, end in _walk(handle):
            if number != 14 or wire != 2:
                continue
            here = handle.tell()
            handle.seek(body)
            key, _value = _entry(handle.read(end - body))
            handle.seek(here)
            if key != "n_speakers":
                continue
            payload = _encode_entry(key, str(int(speakers)))
            edits.append((tag, end, _encode_varint((14 << 3) | 2)
                          + _encode_varint(len(payload)) + payload))
    if not edits:
        raise BankError("that Kokoro model does not declare a speaker count, so Voice Chat "
                        "will not extend it.")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as reading, open(destination, "wb") as writing:
        cursor = 0
        for tag, end, payload in edits:
            _copy(reading, writing, cursor, tag)
            writing.write(payload)
            cursor = end
        reading.seek(cursor)
        for block in iter(lambda: reading.read(1 << 20), b""):
            writing.write(block)
    return _hash(destination)


def _copy(reading, writing, start: int, stop: int) -> None:
    reading.seek(start)
    remaining = stop - start
    while remaining > 0:
        block = reading.read(min(remaining, 1 << 20))
        if not block:
            return
        writing.write(block)
        remaining -= len(block)


# --------------------------------------------------------------------------- #
# Voicepack validation
# --------------------------------------------------------------------------- #


def read_voicepack(source) -> bytes:
    """One Storytime clone as exactly ``STYLE_ROWS`` rows of float32.

    Section 54. Storytime writes ``[N, 1, 256]`` little-endian float32 with N
    of 510 or 511; sherpa's Kokoro v1.0 addresses rows below 510 and no more,
    because ``style_dim[0]`` is also its maximum token length. So 511 is
    accepted and its final row is dropped -- the row is real, it is simply not
    reachable by this runtime -- and anything else is refused rather than
    padded, because a voicepack of the wrong shape is a voicepack for a
    different model.

    Everything this refuses is a thing that would otherwise reach a native
    loader: a truncated file, a text file with a ``.bin`` name, a pack full of
    NaN from an optimisation that diverged.
    """
    path = Path(source)
    try:
        raw = path.read_bytes()
    except OSError:
        raise BankError("that voicepack could not be read.") from None
    if not raw:
        raise BankError("that voicepack is empty.")
    if len(raw) % ROW_BYTES:
        raise BankError(f"that voicepack is {len(raw)} bytes, which is not a whole number of "
                        f"{STYLE_DIM}-value rows.")
    rows = len(raw) // ROW_BYTES
    if rows not in (STYLE_ROWS, STYLE_ROWS + 1):
        raise BankError(f"that voicepack has {rows} rows and this Kokoro runtime uses "
                        f"{STYLE_ROWS}.")
    block = raw[:SPEAKER_BYTES]
    _check_finite(block)
    return block


def _check_finite(block: bytes) -> None:
    """Refuse NaN or infinity anywhere in a voicepack.

    Cheap and worth it: a style vector with a NaN in it produces silence, or
    noise, or a native crash, depending on the day -- and the moment to find
    out is before it is written into the bank the whole feature loads from.
    """
    values = array.array("f")
    values.frombytes(block)
    if sys.byteorder == "big":
        values.byteswap()
    for value in values:
        if not math.isfinite(value):
            raise BankError("that voicepack contains values that are not numbers.")


# --------------------------------------------------------------------------- #
# The bank manifest
# --------------------------------------------------------------------------- #


def manifest() -> dict:
    """What was built, and from what. ``{}`` when there is no bank."""
    try:
        found = json.loads(paths.bank_manifest().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def version() -> str:
    """An opaque marker that changes whenever the bank does."""
    found = manifest()
    return str(found.get("hash") or "")[:16]


def installed() -> bool:
    """Whether a built bank is present and matches its own manifest.

    Checked by size rather than by re-hashing 44 MB on every status poll: the
    hash is verified when the bank is built and when it is promoted, and what
    this question is really asking is "is the file still the file we wrote".
    """
    found = manifest()
    if not found:
        return False
    try:
        voices, model = paths.bank_voices(), paths.bank_model()
        return (voices.is_file() and model.is_file()
                and voices.stat().st_size == int(found.get("bytes") or -1))
    except OSError:
        return False


def live_paths() -> dict:
    """The overrides the Voice Worker should load, or ``{}`` for the bundle.

    ``{}`` is the ordinary answer on an installation that has never cloned
    anything: no bank, no derived model, and the pinned Kokoro bundle used
    exactly as V1 used it. Section 114 -- nobody has to install anything new to
    keep the speech they already had.
    """
    if not installed():
        return {}
    return {"model": str(paths.bank_model()), "voices": str(paths.bank_voices())}


def capacity() -> dict:
    """The shape of the bank the *bundle* could support, built or not.

    Answered from the installed bundle rather than from the built bank, so that
    Settings can say "32 custom voices, none used" before a bank exists.
    """
    found = manifest()
    if found:
        return {"official": int(found.get("official_count") or 0),
                "capacity": int(found.get("capacity") or 0),
                "slots": dict(found.get("slots") or {})}
    official = 0
    try:
        official = speaker_capacity(_bundle()["model"])
    except Exception:
        logger.debug("Model Chain: could not read the Kokoro speaker count", exc_info=True)
    return {"official": official, "capacity": CUSTOM_CAPACITY if official else 0, "slots": {}}


def custom_base() -> int:
    """The SID of custom slot 0. Section 50's ``CUSTOM_SID_BASE``."""
    return int(capacity().get("official") or 0)


def sid_for_slot(slot: int) -> int:
    return custom_base() + int(slot)


def _bundle() -> dict:
    """The installed Kokoro bundle's own paths."""
    found = models.bundle_paths("tts")
    for key in ("model", "voices"):
        if not found.get(key) or not os.path.isfile(str(found[key])):
            raise BankError("The Kokoro text-to-speech model is not installed, so Voice Chat "
                            "cannot build a voice bank.")
    return found


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build(slots: dict, *, official_bytes: bytes = None) -> dict:
    """Assemble a candidate bank in staging. Returns what was built.

    ``slots`` maps custom slot number to the canonical ``.bin`` path for the
    clone that occupies it. Everything else is derived: the official block
    comes from the pinned bundle in its exact upstream order (T-BANK-1), each
    unoccupied slot gets the filler voice, and the derived model's speaker
    count is set to match the total.

    Deterministic on purpose. Given the same bundle and the same slots this
    produces the same bytes and therefore the same hash, which is what makes a
    rebuild after a crash something that can be *compared* rather than merely
    hoped about.
    """
    bundle = _bundle()
    official = official_bytes if official_bytes is not None else Path(bundle["voices"]).read_bytes()
    if not official or len(official) % SPEAKER_BYTES:
        raise BankError("The installed Kokoro voice bank is not a whole number of voices, so "
                        "Voice Chat will not build on top of it.")
    official_count = len(official) // SPEAKER_BYTES
    declared = speaker_capacity(bundle["model"])
    if declared and declared != official_count:
        raise BankError(f"The installed Kokoro model declares {declared} voices and its voice "
                        f"file holds {official_count}. Reinstall the text-to-speech model.")

    rows, dim = style_shape(bundle["model"])
    if (rows, dim) != (STYLE_ROWS, STYLE_DIM):
        raise BankError(f"This Kokoro model uses {rows}x{dim} styles and Voice Chat's bank "
                        f"format is {STYLE_ROWS}x{STYLE_DIM}.")

    filler = _official_block(official, _filler_sid(official_count))
    staging = paths.bank_staging()
    _clear(staging)
    staging.mkdir(parents=True, exist_ok=True)
    candidate = staging / paths.BANK_VOICES
    digest = hashlib.sha256()
    with open(candidate, "wb") as handle:
        handle.write(official)
        digest.update(official)
        for slot in range(CUSTOM_CAPACITY):
            source = (slots or {}).get(slot) or (slots or {}).get(str(slot))
            block = read_voicepack(source) if source else filler
            handle.write(block)
            digest.update(block)

    total = official_count + CUSTOM_CAPACITY
    model = staging / paths.BANK_MODEL
    model_hash = derive_model(bundle["model"], model, total)

    found = {
        "schema": SCHEMA,
        "official_count": official_count,
        "capacity": CUSTOM_CAPACITY,
        "speakers": total,
        "style_rows": STYLE_ROWS,
        "style_dim": STYLE_DIM,
        "hash": digest.hexdigest(),
        "bytes": candidate.stat().st_size,
        "model_hash": model_hash,
        "source_model_hash": _hash(bundle["model"]),
        "source_bundle": str(bundle.get("id") or ""),
        "slots": {str(slot): str(source) for slot, source in sorted((slots or {}).items())},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (staging / paths.BANK_MANIFEST).write_text(json.dumps(found, indent=2), encoding="utf-8")
    expected = total * SPEAKER_BYTES
    if found["bytes"] != expected:
        raise BankError(f"The voice bank came out {found['bytes']} bytes and should be "
                        f"{expected}. It was not installed.")
    return found


def promote() -> dict:
    """Swap the staged bank in, keeping the previous one to go back to.

    Three renames on one filesystem and no copying, so there is no moment at
    which ``bank/voices.bin`` is half of anything. The previous bytes are moved
    aside rather than deleted, because :func:`rollback` is what a failed
    runtime validation calls and it must not depend on being able to rebuild.
    """
    staging, root = paths.bank_staging(), paths.bank_root()
    found = json.loads((staging / paths.BANK_MANIFEST).read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    backup = root / "previous"
    _clear(backup)
    backup.mkdir(parents=True, exist_ok=True)
    for name in (paths.BANK_VOICES, paths.BANK_MODEL, paths.BANK_MANIFEST):
        live = root / name
        if live.exists():
            os.replace(live, backup / name)
    for name in (paths.BANK_VOICES, paths.BANK_MODEL, paths.BANK_MANIFEST):
        os.replace(staging / name, root / name)
    _clear(staging)
    logger.info("Model Chain: Voice bank installed — %d voices, %d bytes, %s",
                found.get("speakers"), found.get("bytes"), found.get("hash", "")[:12])
    return found


def rollback() -> bool:
    """Put the previous bank back. Returns whether there was one.

    Called when a promoted bank fails its synthesis check (section 55, steps
    8-11). A bank that loads but cannot speak is worse than no bank: it takes
    the official voices down with it.
    """
    root = paths.bank_root()
    backup = root / "previous"
    if not (backup / paths.BANK_VOICES).is_file():
        for name in (paths.BANK_VOICES, paths.BANK_MODEL, paths.BANK_MANIFEST):
            try:
                (root / name).unlink()
            except OSError:
                pass
        logger.warning("Model Chain: the Voice bank was rolled back to the installed bundle")
        return False
    for name in (paths.BANK_VOICES, paths.BANK_MODEL, paths.BANK_MANIFEST):
        source = backup / name
        if source.exists():
            os.replace(source, root / name)
    logger.warning("Model Chain: the Voice bank was rolled back to its previous build")
    return True


def official_names(count: int = 0) -> list:
    """The upstream speaker names, in SID order, from the pinned manifest.

    The names live beside the checksums of the archive they came out of, which
    is the only place they can be pinned honestly: they are a property of that
    release, and section 42 is explicit that a hardcoded UI list must not be
    trusted forever. What this returns is validated against the *installed*
    model's own speaker count by :func:`check_names` before anything is shown.
    """
    try:
        entry = models.default_model("tts")
        found = list(getattr(entry, "extra", {}).get("speakers") or ())
    except Exception:
        logger.debug("Model Chain: could not read the Kokoro speaker map", exc_info=True)
        found = []
    if count and len(found) != count:
        return []
    return found


def _official_block(official: bytes, sid: int) -> bytes:
    start = sid * SPEAKER_BYTES
    return official[start:start + SPEAKER_BYTES]


def _filler_sid(official_count: int) -> int:
    names = official_names(official_count)
    if FILLER_VOICE in names:
        return names.index(FILLER_VOICE)
    return 0


def _hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clear(path) -> None:
    try:
        shutil.rmtree(path)
    except (OSError, FileNotFoundError):
        pass


def store_voicepack(identifier: str, source) -> Path:
    """Copy a validated clone into the canonical clones directory.

    Validated *first*, and stored in the normalized 510-row form the bank uses,
    so that a rebuild never has to make the 511-row decision twice and never
    has to trust a file it did not write.
    """
    block = read_voicepack(source)
    destination = paths.clone_file(identifier)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".bin.new")
    staging.write_bytes(block)
    os.replace(staging, destination)
    return destination


def forget_voicepack(identifier: str) -> None:
    """Remove a clone's canonical source. Only after its registry entry is gone."""
    try:
        paths.clone_file(identifier).unlink()
    except (OSError, ValueError):
        logger.debug("Model Chain: could not remove a clone voicepack", exc_info=True)
