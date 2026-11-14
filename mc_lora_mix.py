"""Mixing two LoRAs into one new LoRA file.

Asked for: "an advanced setting ... where i can mix lora ... choose two lora by
typing out and prefill (just like extra networks), i can choose their strength,
and we can mix them into a single new lora that i can put into a choose
directory in the lora directory. I should be forced to choose output folder
everytime. I can also choose to name it something different. Once the lora is
created, our extension should remember it was made by the extension, i would
like a placed holder image until i set one myself." (docs/26 §10.)

**The mix is exact.** Forge applies a LoRA layer as ``strength * alpha / rank *
up @ down`` (its vendored ComfyUI ``weight_adapter/lora.py``). Two LoRAs at
strengths ``s_a`` and ``s_b`` change a layer both touch by the sum of their two
products, and that sum *is* one LoRA of the two ranks together: the down
matrices stacked, the up matrices side by side, each up scaled by its own
``c = s * alpha / rank``. To keep the stored numbers as they were, the larger
``|c|`` goes into the new alpha (``alpha = k * (rank_a + rank_b)``, ``k`` the
larger) and only the other LoRA's up matrix is scaled, by ``c / k``; a layer
only one LoRA touches keeps its matrices and gets ``alpha = |c| * rank`` (a
negative strength flips its up matrix's sign, which is exact). A ``.diff`` style
patch (and Forge's ``diff_b``, ``w_norm``, ``b_norm``) is summed at the two
strengths. A LoCon Tucker ``lora_mid`` is folded into its down matrix exactly
as Forge folds it. Nothing is approximated, so nothing needs a rank to be
chosen: the new file's rank is the two together, its size about theirs
together. What cannot be mixed exactly is refused, by name: DoRA (its magnitude
is normalised against the whole weight), LoHa, LoKr, OFT, GLoRA, a weight
replacement.

**One layer, whichever way it is spelled.** Trainers name a layer differently
(``lora_unet_layers_0_attn_qkv``, ``diffusion_model.layers.0.attn.qkv``, bare
``layers.0.attn.qkv``), and Forge maps every spelling of one layer to the same
weight -- where two names in one file reach the same weight the second
overwrites the first. So layers are matched by one canonical name (the known
prefixes off, dots as underscores) and the mix writes each layer once, in the
first LoRA's spelling. A layer two LoRAs change with different sizes means
they were not made for the same model: refused, naming the layer. Two LoRAs
that share no layer and name their layers in different schemes are refused
too, since Forge might apply only one of them wherever both change a layer.

**The file.** Read and written here with numpy, not torch or ``safetensors``:
half, bfloat16 and single precision, the dtypes LoRAs are saved in (bfloat16
by its bits, rounded to nearest even). Each layer keeps its precision when
both LoRAs share it; half and bfloat16 together are written as bfloat16, and
single precision wins over either. The new file's metadata is its own:
``ss_output_name`` its name (Forge reads a LoRA's alias from it, so a copied
one would have named the mix after its first LoRA), ``modelspec.title``, and
``mc_mix``, a JSON record of what it was made from -- the two files, their
SHA256 and strengths -- which is how it is known as a mix wherever it is moved.
The inputs' own ``sshs_model_hash`` is never carried (Forge would take it for
the mix's hash).

**Where it goes.** A folder under Forge's LoRA folders, chosen every time
(nothing is remembered), named as asked or ``mix_<first>_<second>``, never over
a file that is there and never under a name Forge already knows. Written to a
temporary file beside it and linked into place, so a half-written mix never
has the name.

**What is remembered.** ``mixes.json`` in the LoRA data root, one entry per mix
(the file, when, from what, its placeholder), shown on the tab's Mix page; the
file's own ``mc_mix`` record, shown in the LoRA's details dialog; and Forge's
user metadata beside it (``<name>.json``, written only when there is none): a
description that says it is a mix, and both LoRAs' activation texts.

**The placeholder.** A drawn card -- the two LoRAs as two overlapping discs,
each as large as its strength -- saved as ``<name>.<samples_format>``, which is
the very file Forge's *Replace preview* writes, so setting a picture of your
own replaces it in place. Forge looks for a card's picture in a *set* of
extensions, in no fixed order, so a picture of yours under another extension
would race the placeholder: :func:`sweep_placeholders` deletes a placeholder
still as it was drawn whenever another picture is beside it (after a mix, when
the Mix page lists the mixes, and at every start).

Log lines name no file, folder or LoRA (the standing convention); the job's
own lines, shown only on the page, name the LoRAs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import struct
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import mc_lora_paths
from mc_lora_common import lora_directories, log, opt, report

MIXES_FILENAME = "mixes.json"
METADATA_KEY = "mc_mix"
MADE_BY = "Model Chain LoRA mixer"
RECORD_FORMAT = 1
STRENGTH_MIN = -5.0
STRENGTH_MAX = 5.0
MAX_HEADER_BYTES = 100_000_000
MAX_NAME_LENGTH = 150
MAX_FOLDERS = 2000
PLACEHOLDER_SIZE = (512, 768)
# Forge's ``allowed_preview_extensions`` (modules/ui_extra_networks.py), a set:
# any of these beside a LoRA, plain or ``.preview.``, is its card's picture.
PREVIEW_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "jxl", "avif", "heif", "gif", "mp4", "webm")
PLACEHOLDER_FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}
# Kept from the inputs when every input that has one agrees: what a model
# browser reads a LoRA's base model by.
CARRIED_METADATA = ("modelspec.architecture", "modelspec.implementation", "ss_base_model_version")

_REGISTRY_LOCK = threading.Lock()


class MixError(ValueError):
    """A mix refused, with the reason in words for the person."""


# --------------------------------------------------------------------------- #
# safetensors, read and written with numpy
# --------------------------------------------------------------------------- #

ITEM_SIZES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
FLOAT_DTYPES = ("F16", "BF16", "F32", "F64")


@dataclass
class Tensor:
    dtype: str
    shape: tuple
    data: memoryview

    def floats(self) -> np.ndarray:
        """The values as float32, in their shape."""
        if self.dtype == "BF16":
            bits = np.frombuffer(self.data, dtype="<u2").astype(np.uint32) << 16
            values = bits.view(np.float32)
        elif self.dtype == "F16":
            values = np.frombuffer(self.data, dtype="<f2").astype(np.float32)
        elif self.dtype == "F32":
            values = np.frombuffer(self.data, dtype="<f4").astype(np.float32)
        elif self.dtype == "F64":
            values = np.frombuffer(self.data, dtype="<f8").astype(np.float32)
        else:
            raise MixError(f"a tensor is stored as {self.dtype}, which is not a weight this mixer reads")
        return values.reshape(self.shape)

    def scalar(self) -> float:
        values = self.floats().reshape(-1)
        if values.size != 1:
            raise MixError("an alpha holds more than one number")
        return float(values[0])


def read_safetensors(path) -> tuple[dict, dict[str, Tensor]]:
    """A safetensors file's metadata and tensors (their bytes held in memory).
    Raises :class:`MixError` for anything that is not one."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                raise MixError(f"{path.name} is not a safetensors file")
            length = struct.unpack("<Q", head)[0]
            if length <= 1 or length > MAX_HEADER_BYTES:
                raise MixError(f"{path.name} is not a safetensors file")
            header = json.loads(handle.read(length).decode("utf-8"))
            body = handle.read()
    except OSError as exc:
        raise MixError(f"{path.name} could not be read ({exc.strerror or type(exc).__name__})") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise MixError(f"{path.name} is not a safetensors file") from None
    if not isinstance(header, dict):
        raise MixError(f"{path.name} is not a safetensors file")

    metadata = header.pop("__metadata__", None) or {}
    view = memoryview(body)
    tensors: dict[str, Tensor] = {}
    for name, info in header.items():
        try:
            dtype = str(info["dtype"])
            shape = tuple(int(size) for size in info["shape"])
            start, end = (int(offset) for offset in info["data_offsets"])
        except (KeyError, TypeError, ValueError):
            raise MixError(f"{path.name} has a tensor its header does not describe") from None
        count = 1
        for size in shape:
            count *= size
        if dtype not in ITEM_SIZES or not (0 <= start <= end <= len(body)) or end - start != count * ITEM_SIZES[dtype]:
            raise MixError(f"{path.name} has a tensor that does not match its header")
        tensors[name] = Tensor(dtype, shape, view[start:end])
    return {str(key): str(value) for key, value in metadata.items()}, tensors


def encode(values: np.ndarray, dtype: str) -> bytes:
    """float32 values as the bytes of ``dtype``; bfloat16 rounded to nearest even."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if dtype == "F32":
        return values.astype("<f4").tobytes()
    if dtype == "F16":
        out = values.astype("<f2")
        if not np.isfinite(out).all() and np.isfinite(values).all():
            raise MixError("a mixed weight is too large for half precision; lower the strengths")
        return out.tobytes()
    if dtype == "BF16":
        bits = values.view(np.uint32)
        rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16).astype("<u2")
        nan = np.isnan(values)
        if nan.any():
            rounded[nan] = 0x7FC0
        return rounded.tobytes()
    raise MixError(f"cannot write {dtype}")


@dataclass
class OutTensor:
    name: str
    dtype: str
    shape: tuple
    data: bytes


def write_safetensors(path, metadata: dict, tensors: list[OutTensor]) -> str:
    """Write a safetensors file; the SHA256 of what was written."""
    header: dict = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    offset = 0
    for tensor in tensors:
        header[tensor.name] = {"dtype": tensor.dtype, "shape": list(tensor.shape),
                               "data_offsets": [offset, offset + len(tensor.data)]}
        offset += len(tensor.data)
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    digest = hashlib.sha256()
    with Path(path).open("wb") as handle:
        for chunk in (struct.pack("<Q", len(raw)), raw, *(tensor.data for tensor in tensors)):
            handle.write(chunk)
            digest.update(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# A LoRA's layers
# --------------------------------------------------------------------------- #

# Every spelling of a layer's parts Forge's LoRA adapter reads, longest first so
# ".lora_A.default.weight" is never read as ".lora_A".
ROLE_SUFFIXES = tuple(sorted((
    (".lora_up.weight", "up"), (".lora_down.weight", "down"), (".lora_mid.weight", "mid"),
    ("_lora.up.weight", "up"), ("_lora.down.weight", "down"),
    (".lora_B.weight", "up"), (".lora_A.weight", "down"),
    (".lora.up.weight", "up"), (".lora.down.weight", "down"),
    (".lora_linear_layer.up.weight", "up"), (".lora_linear_layer.down.weight", "down"),
    (".lora_B.default.weight", "up"), (".lora_A.default.weight", "down"),
    (".lora_B", "up"), (".lora_A", "down"),
    (".alpha", "alpha"),
    (".diff_b", "diff_b"), (".diff", "diff"), (".w_norm", "w_norm"), (".b_norm", "b_norm"),
), key=lambda pair: -len(pair[0])))

DIFF_ROLES = ("diff", "diff_b", "w_norm", "b_norm")

# What cannot be mixed exactly, by the mark a tensor's name carries.
UNMIXABLE = (
    (".dora_scale", "DoRA"),
    (".hada_", "LoHa"),
    (".lokr_", "LoKr"),
    (".oft_", "OFT"),
    (".boft_", "BOFT"),
    (".set_weight", "a weight replacement"),
    (".reshape_weight", "a reshape"),
    (".a1.weight", "GLoRA"), (".a2.weight", "GLoRA"), (".b1.weight", "GLoRA"), (".b2.weight", "GLoRA"),
)

# The prefixes Forge maps to the diffusion model, whichever a trainer wrote.
LAYER_HEADS = ("lora_unet_", "lycoris_", "model.diffusion_model.", "diffusion_model.", "transformer.", "unet.")


def canonical(prefix: str) -> str:
    """One name for a layer however it is spelled: the diffusion model's prefix
    off, dots as underscores."""
    name = prefix
    for head in LAYER_HEADS:
        if name.startswith(head):
            name = name[len(head):]
            break
    return name.replace(".", "_")


def scheme(prefix: str) -> str:
    """How a layer's name is spelled: its prefix, or bare."""
    for head in LAYER_HEADS:
        if prefix.startswith(head):
            return head
    if prefix.startswith("lora_te"):
        return "lora_te"
    return "bare-dotted" if "." in prefix else "bare"


@dataclass
class Layer:
    prefix: str
    parts: dict = field(default_factory=dict)

    def lora(self) -> tuple[np.ndarray, np.ndarray, float, list[str]] | None:
        """``(down, up, alpha, dtypes)``, the Tucker middle folded into down as
        Forge folds it; None for a layer that is no low-rank pair."""
        if "up" not in self.parts:
            return None
        down = self.parts["down"].floats()
        up = self.parts["up"].floats()
        dtypes = [self.parts["up"].dtype, self.parts["down"].dtype]
        if "mid" in self.parts:
            mid = self.parts["mid"].floats()
            dtypes.append(self.parts["mid"].dtype)
            if down.ndim != 4 or mid.ndim != 4:
                raise MixError(f"layer {canonical(self.prefix)} has a middle matrix in a shape this mixer does not read")
            final = (down.shape[1], down.shape[0], mid.shape[2], mid.shape[3])
            folded = np.transpose(down, (1, 0, 2, 3)).reshape(down.shape[1], -1) @ \
                np.transpose(mid, (1, 0, 2, 3)).reshape(mid.shape[1], -1)
            down = np.transpose(folded.reshape(final), (1, 0, 2, 3))
        rank = down.shape[0]
        if up.ndim < 2 or up.shape[1] != rank:
            raise MixError(f"layer {canonical(self.prefix)}'s two matrices do not fit together")
        alpha = self.parts["alpha"].scalar() if "alpha" in self.parts else float(rank)
        return down, up, alpha, dtypes


def layers_of(tensors: dict[str, Tensor], label: str) -> dict[str, Layer]:
    """A LoRA's layers by canonical name, in the file's order. Raises for what
    cannot be mixed, naming the LoRA (``label``) and the format."""
    by_prefix: dict[str, Layer] = {}
    unknown: list[str] = []
    for key, tensor in tensors.items():
        for mark, kind in UNMIXABLE:
            if mark in key:
                raise MixError(f"{label} is {kind}, which cannot be mixed exactly")
        for suffix, role in ROLE_SUFFIXES:
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                layer = by_prefix.setdefault(prefix, Layer(prefix))
                if role in layer.parts:
                    raise MixError(f"{label} names layer {canonical(prefix)} twice")
                layer.parts[role] = tensor
                break
        else:
            unknown.append(key)
    if unknown:
        raise MixError(f"{label} holds {len(unknown)} tensor(s) of a kind this mixer does not read "
                       f"(the first: {unknown[0]})")

    layers: dict[str, Layer] = {}
    for prefix, layer in by_prefix.items():
        roles = set(layer.parts)
        if ("up" in roles) != ("down" in roles):
            raise MixError(f"{label}'s layer {canonical(prefix)} has one of its two matrices only")
        if "mid" in roles and "up" not in roles:
            raise MixError(f"{label}'s layer {canonical(prefix)} has a middle matrix and nothing around it")
        if roles == {"alpha"}:
            continue  # an alpha with no matrices: Forge ignores it too
        name = canonical(prefix)
        if name in layers:
            raise MixError(f"{label} names layer {name} twice")
        layers[name] = layer
    if not layers:
        raise MixError(f"{label} holds no LoRA layers")
    return layers


# --------------------------------------------------------------------------- #
# The mix
# --------------------------------------------------------------------------- #


def promote(dtypes: Iterable[str]) -> str:
    """One precision for a layer from two: single wins, then one shared, and
    half beside bfloat16 is bfloat16 (the range a mix can need)."""
    found = set(dtypes)
    if found & {"F32", "F64"}:
        return "F32"
    if len(found) == 1:
        return found.pop()
    return "BF16"


@dataclass
class Source:
    """One LoRA to mix: its label for messages, its tensors and metadata, and
    the strength it is mixed at."""

    label: str
    tensors: dict
    metadata: dict
    strength: float
    layers: dict = field(default_factory=dict)


@dataclass
class Outcome:
    tensors: list
    shared: int = 0
    only_first: int = 0
    only_second: int = 0
    rank: int = 0


def mix(first: Source, second: Source, *, cancelled: Callable[[], bool] = lambda: False) -> Outcome:
    """The two LoRAs as one: every layer either changes, at the strengths given."""
    first.layers = layers_of(first.tensors, first.label)
    second.layers = layers_of(second.tensors, second.label)
    shared = [name for name in first.layers if name in second.layers]
    if not shared:
        if not {scheme(layer.prefix) for layer in first.layers.values()} & \
                {scheme(layer.prefix) for layer in second.layers.values()}:
            raise MixError(f"{first.label} and {second.label} share no layer and name their layers differently: "
                           "they were made for different models, or by tools Forge would read as changing the "
                           "same layer twice")

    outcome = Outcome(tensors=[], shared=len(shared),
                      only_first=len(first.layers) - len(shared),
                      only_second=len(second.layers) - len(shared))
    order = list(first.layers) + [name for name in second.layers if name not in first.layers]
    for name in order:
        if cancelled():
            raise MixError("cancelled")
        parts = [(source.layers[name], source.strength) for source in (first, second)
                 if name in source.layers and source.strength != 0]
        if not parts:
            continue
        prefix = parts[0][0].prefix
        rank = _mix_lora(name, prefix, parts, outcome)
        outcome.rank = max(outcome.rank, rank)
        for role in DIFF_ROLES:
            _mix_diff(name, prefix, role, [(layer, strength) for layer, strength in parts if role in layer.parts],
                      outcome)
    if not outcome.tensors:
        raise MixError("nothing to write: both strengths leave every layer unchanged")
    return outcome


def _mix_lora(name: str, prefix: str, parts, outcome: Outcome) -> int:
    pieces = []
    for layer, strength in parts:
        found = layer.lora()
        if found is None:
            continue
        down, up, alpha, dtypes = found
        scale = strength * alpha / down.shape[0]
        if scale != 0:
            pieces.append((down, up, scale, dtypes))
    if not pieces:
        return 0
    if len(pieces) == 2:
        (down_a, up_a, _, _), (down_b, up_b, _, _) = pieces
        if down_a.shape[1:] != down_b.shape[1:] or up_a.shape[0] != up_b.shape[0] or up_a.shape[2:] != up_b.shape[2:]:
            raise MixError(f"layer {name} is a different size in the two LoRAs: they were not made for the same model")
    largest = max(abs(scale) for _, _, scale, _ in pieces)
    downs = [down for down, _, _, _ in pieces]
    ups = [up * np.float32(scale / largest) if scale != largest else up for _, up, scale, _ in pieces]
    rank = sum(down.shape[0] for down in downs)
    dtype = promote(dtype for _, _, _, dtypes in pieces for dtype in dtypes)
    down = downs[0] if len(downs) == 1 else np.concatenate(downs, axis=0)
    up = ups[0] if len(ups) == 1 else np.concatenate(ups, axis=1)
    outcome.tensors.append(OutTensor(f"{prefix}.lora_down.weight", dtype, down.shape, encode(down, dtype)))
    outcome.tensors.append(OutTensor(f"{prefix}.lora_up.weight", dtype, up.shape, encode(up, dtype)))
    outcome.tensors.append(OutTensor(f"{prefix}.alpha", "F32", (), encode(np.array(largest * rank), "F32")))
    return rank


def _mix_diff(name: str, prefix: str, role: str, parts, outcome: Outcome) -> None:
    if not parts:
        return
    shapes = {layer.parts[role].shape for layer, _ in parts}
    if len(shapes) != 1:
        raise MixError(f"layer {name} is a different size in the two LoRAs: they were not made for the same model")
    total = None
    for layer, strength in parts:
        values = layer.parts[role].floats() * np.float32(strength)
        total = values if total is None else total + values
    dtype = promote(layer.parts[role].dtype for layer, _ in parts)
    outcome.tensors.append(OutTensor(f"{prefix}.{role}", dtype, total.shape, encode(total, dtype)))


def carried(first: dict, second: dict) -> dict:
    """The inputs' base-model keys every input that has one agrees on."""
    out = {}
    for key in CARRIED_METADATA:
        values = {meta[key] for meta in (first, second) if meta.get(key)}
        if len(values) == 1:
            out[key] = values.pop()
    return out


# --------------------------------------------------------------------------- #
# Names and folders
# --------------------------------------------------------------------------- #

_FORBIDDEN = set('<>:"/\\|?*')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def default_name(first: str, second: str) -> str:
    return f"mix_{first}_{second}"


def check_name(raw: str, first: str, second: str) -> str:
    """The new LoRA's name as asked, or the default when nothing is asked; a
    name no folder can hold is refused, never altered."""
    name = str(raw or "").strip()
    if name.lower().endswith(".safetensors"):
        name = name[: -len(".safetensors")].rstrip()
    if not name:
        name = "".join("_" if char in _FORBIDDEN or ord(char) < 32 else char
                       for char in default_name(first, second))
    bad = sorted({char for char in name if char in _FORBIDDEN or ord(char) < 32})
    if bad:
        shown = " ".join(repr(char) if ord(char) < 32 else char for char in bad)
        raise MixError(f"a LoRA's name cannot hold {shown}")
    if name in (".", "..") or name.endswith((".", " ")) or name.split(".")[0].lower() in _RESERVED:
        raise MixError(f"“{name}” cannot be a file's name")
    if len(name) > MAX_NAME_LENGTH:
        raise MixError(f"the name is longer than {MAX_NAME_LENGTH} characters")
    return name


def folders(roots: Iterable[str] | None = None) -> list[tuple[str, Path]]:
    """``(label, folder)`` for every LoRA folder and every folder under it,
    hidden ones left out: what the output folder is chosen from. The label is
    the folder under its LoRA folder, which is what is shown and sent back."""
    roots = [Path(root) for root in (roots if roots is not None else lora_directories())]
    roots = [root for root in roots if root.is_dir()]
    found: list[tuple[str, Path]] = []
    used: set[str] = set()
    for index, root in enumerate(roots):
        head = "LoRA folder" if len(roots) == 1 else (root.name or f"LoRA folder {index + 1}")
        while head in used:
            head = f"{head} ({index + 1})"
        used.add(head)
        found.append((head, root))
        below: list[tuple[str, Path]] = []
        for current, names, _files in os.walk(root):
            names[:] = sorted((entry for entry in names if not entry.startswith(".")), key=str.casefold)
            for entry in names:
                path = Path(current) / entry
                relative = path.relative_to(root).as_posix()
                below.append((relative if len(roots) == 1 else f"{head}/{relative}", path))
                if len(found) + len(below) >= MAX_FOLDERS:
                    break
            if len(found) + len(below) >= MAX_FOLDERS:
                break
        found.extend(sorted(below, key=lambda pair: pair[0].casefold()))
    return found


def folder_for(label: str, roots: Iterable[str] | None = None) -> Path | None:
    """The folder a label names, looked up afresh: only a folder the list holds."""
    return next((path for shown, path in folders(roots) if shown == label), None)


# --------------------------------------------------------------------------- #
# Forge's LoRAs
# --------------------------------------------------------------------------- #


def forge_networks():
    import mc_lora_viewer

    return mc_lora_viewer.forge_networks()


def loras(networks=None, roots: Iterable[str] | None = None) -> list[tuple[str, str]]:
    """``(label, name)`` of every safetensors LoRA Forge knows, the label its
    place under its LoRA folder: what the two pickers offer, filtered as typed."""
    networks = networks if networks is not None else forge_networks()
    if networks is None:
        return []
    roots = [Path(root) for root in (roots if roots is not None else lora_directories())]
    found = []
    for name, entry in list((getattr(networks, "available_networks", None) or {}).items()):
        filename = str(getattr(entry, "filename", "") or "")
        if not filename.lower().endswith(".safetensors"):
            continue
        path = Path(filename)
        label = path.stem
        for root in roots:
            try:
                label = path.relative_to(root).with_suffix("").as_posix()
                break
            except ValueError:
                continue
        found.append((label, str(name)))
    found.sort(key=lambda pair: (pair[0].casefold(), pair[0]))
    return found


def lora_file(name: str, networks=None) -> Path | None:
    networks = networks if networks is not None else forge_networks()
    entry = (getattr(networks, "available_networks", None) or {}).get(name) if networks is not None else None
    filename = str(getattr(entry, "filename", "") or "") if entry is not None else ""
    return Path(filename) if filename else None


def known_name(name: str, networks=None) -> bool:
    """Whether Forge already knows a LoRA by this name or alias."""
    networks = networks if networks is not None else forge_networks()
    if networks is None:
        return False
    wanted = name.casefold()
    for table in ("available_networks", "available_network_aliases"):
        if any(str(key).casefold() == wanted for key in (getattr(networks, table, None) or {})):
            return True
    return False


def refresh_forge(name: str, networks=None) -> bool:
    """Tell Forge's LoRA list about the new file, as its own lookup of a name it
    does not know yet does; the extra-network browser shows its card after its
    own refresh."""
    networks = networks if networks is not None else forge_networks()
    if networks is None:
        return False
    try:
        if hasattr(networks, "update_available_networks_by_names"):
            networks.update_available_networks_by_names([name])
        elif hasattr(networks, "list_available_networks"):
            networks.list_available_networks()
        else:
            return False
    except Exception:
        report("could not tell Forge about a new LoRA mix")
        return False
    return True


# --------------------------------------------------------------------------- #
# What is remembered
# --------------------------------------------------------------------------- #


def registry_path() -> Path:
    return mc_lora_paths.data_root() / MIXES_FILENAME


def load_registry() -> list[dict]:
    try:
        payload = json.loads(registry_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        report("could not read the list of LoRA mixes")
        return []
    entries = payload.get("mixes") if isinstance(payload, dict) else None
    return [entry for entry in entries or [] if isinstance(entry, dict)]


def _save_registry(entries: list[dict]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".mixes-",
                                         suffix=".tmp", delete=False) as handle:
            temp = handle.name
            json.dump({"version": RECORD_FORMAT, "mixes": entries}, handle, indent=2, ensure_ascii=False)
        os.replace(temp, path)
        temp = None
    finally:
        if temp:
            try:
                os.unlink(temp)
            except OSError:
                pass


def remember(entry: dict) -> None:
    with _REGISTRY_LOCK:
        entries = load_registry()
        entries.append(entry)
        _save_registry(entries)


def mix_record(metadata) -> dict | None:
    """What a LoRA's own metadata says it was mixed from, or None."""
    raw = (metadata or {}).get(METADATA_KEY) if isinstance(metadata, dict) else None
    if not raw:
        return None
    try:
        record = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    if not isinstance(record, dict) or record.get("made_by") != MADE_BY:
        return None
    return record


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def other_previews(mix_path: Path, ignore: Path | None = None) -> list[Path]:
    """Every picture Forge would take for the LoRA's card, but ``ignore``."""
    stem = mix_path.with_suffix("")
    found = []
    for ext in PREVIEW_EXTENSIONS:
        for candidate in (Path(f"{stem}.{ext}"), Path(f"{stem}.preview.{ext}")):
            if candidate.is_file() and (ignore is None or candidate != ignore):
                found.append(candidate)
    return found


def picture_state(entry: dict) -> str:
    """``placeholder`` while the drawn card is the LoRA's picture, ``yours``
    once another is, ``none`` when it has none."""
    path = Path(entry.get("path", ""))
    placeholder = entry.get("placeholder") or {}
    drawn = path.parent / placeholder["file"] if placeholder.get("file") else None
    if drawn is not None and drawn.is_file():
        try:
            if _file_digest(drawn) == placeholder.get("sha256"):
                return "placeholder"
        except OSError:
            pass
        return "yours"
    return "yours" if other_previews(path) else "none"


def sweep_placeholders() -> int:
    """Delete each placeholder that is still as it was drawn while another
    picture is beside its LoRA: Forge picks a card's picture from a set of
    extensions in no fixed order, so the two would race. How many went."""
    removed = 0
    with _REGISTRY_LOCK:
        entries = load_registry()
        changed = False
        for entry in entries:
            placeholder = entry.get("placeholder") or {}
            if not placeholder.get("file"):
                continue
            path = Path(entry.get("path", ""))
            drawn = path.parent / placeholder["file"]
            try:
                if not drawn.is_file() or _file_digest(drawn) != placeholder.get("sha256"):
                    continue
                if other_previews(path, ignore=drawn):
                    drawn.unlink()
                    entry["placeholder"] = None
                    changed = True
                    removed += 1
            except OSError:
                continue
        if changed:
            _save_registry(entries)
    if removed:
        log(f"removed {removed} mix placeholder picture(s): a picture of your own was beside each")
    return removed


def mixes() -> list[dict]:
    """The mixes made here, newest first, each with whether its file is still
    where it was made and what its picture is."""
    out = []
    for entry in reversed(load_registry()):
        shown = dict(entry)
        path = Path(entry.get("path", ""))
        shown["present"] = path.is_file()
        shown["picture"] = picture_state(entry) if shown["present"] else "none"
        out.append(shown)
    return out


# --------------------------------------------------------------------------- #
# The placeholder and the user metadata
# --------------------------------------------------------------------------- #


def placeholder_extension() -> str:
    """Forge's *Images format*, which its Replace preview writes in, so a
    picture you set replaces the placeholder in place."""
    wanted = str(opt("samples_format", "png") or "png").lower().lstrip(".")
    return wanted if wanted in PLACEHOLDER_FORMATS else "png"


def _hue(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:6], 16) / 0xFFFFFF


def draw_placeholder(name: str, inputs: list[tuple[str, float]]):
    """The drawn card: each LoRA a disc as large as its strength, the two
    overlapping, the new LoRA's name under them."""
    import colorsys

    from PIL import Image, ImageDraw, ImageFont

    width, height = PLACEHOLDER_SIZE
    card = Image.new("RGB", (width, height), (24, 26, 32))
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    largest = max((abs(strength) for _, strength in inputs), default=1.0) or 1.0
    centres = ((width * 0.38, height * 0.4), (width * 0.62, height * 0.4))
    for (label, strength), (cx, cy) in zip(inputs, centres):
        radius = width * (0.16 + 0.14 * abs(strength) / largest)
        red, green, blue = (int(channel * 255) for channel in colorsys.hsv_to_rgb(_hue(label), 0.55, 0.95))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(red, green, blue, 150),
                     outline=(red, green, blue, 230), width=4)
    card.paste(layer, (0, 0), layer)
    text = ImageDraw.Draw(card)

    def font(size):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow before 10.1: one small bitmap font
            return ImageFont.load_default()

    lines = [(name, font(30), (236, 238, 242))]
    lines += [(f"{label}  ×{strength:g}", font(20), (170, 176, 188)) for label, strength in inputs]
    lines.append(("LoRA mix", font(18), (120, 126, 138)))
    y = height * 0.68
    for line, face, colour in lines:
        shown = line if len(line) <= 34 else line[:33] + "…"
        box = text.textbbox((0, 0), shown, font=face)
        text.text(((width - (box[2] - box[0])) / 2, y), shown, font=face, fill=colour)
        y += (box[3] - box[1]) + 14
    return card


def write_placeholder(mix_path: Path, name: str, inputs: list[tuple[str, float]]) -> dict | None:
    """The placeholder beside the new LoRA, unless a picture is there already:
    ``{"file", "sha256"}`` or None."""
    if other_previews(mix_path):
        return None
    extension = placeholder_extension()
    target = mix_path.with_suffix(f".{extension}")
    try:
        image = draw_placeholder(name, inputs)
        image.save(target, PLACEHOLDER_FORMATS[extension])
        return {"file": target.name, "sha256": _file_digest(target)}
    except Exception:
        report("could not draw a LoRA mix's placeholder picture")
        return None


def _user_metadata(path: Path) -> dict:
    try:
        payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _joined(*texts: str) -> str:
    words: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for word in re.split(r"\s*,\s*", str(text or "").strip()):
            if word and word.casefold() not in seen:
                seen.add(word.casefold())
                words.append(word)
    return ", ".join(words)


def write_user_metadata(mix_path: Path, sources: list[tuple[Path, str, float]]) -> bool:
    """Forge's user metadata for the new LoRA (``<name>.json``), when it has
    none: a description saying what it was mixed from, both LoRAs' activation
    and negative texts, and their base model where they agree."""
    target = mix_path.with_suffix(".json")
    if target.exists():
        return False
    own = [_user_metadata(path) for path, _, _ in sources]
    versions = {str(meta.get("sd version") or "Unknown") for meta in own} - {"Unknown", ""}
    described = " and ".join(f"{label} at {strength:g}" for _, label, strength in sources)
    payload = {
        "description": f"Mixed by Model Chain from {described}.",
        "sd version": versions.pop() if len(versions) == 1 else "Unknown",
        "activation text": _joined(*(meta.get("activation text", "") for meta in own)),
        "preferred weight": 0.0,
        "negative text": _joined(*(meta.get("negative text", "") for meta in own)),
        "notes": f"Made by {MADE_BY}.",
    }
    try:
        target.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    except OSError:
        report("could not write a LoRA mix's description")
        return False
    return True


# --------------------------------------------------------------------------- #
# A mix, start to end
# --------------------------------------------------------------------------- #


@dataclass
class Request:
    """A mix asked for, checked: what the job needs."""

    first: str
    first_path: Path
    first_strength: float
    second: str
    second_path: Path
    second_strength: float
    folder_label: str
    folder: Path
    name: str

    @property
    def target(self) -> Path:
        return self.folder / f"{self.name}.safetensors"


def _strength(value, which: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise MixError(f"the {which} LoRA's strength is not a number") from None
    if not np.isfinite(number) or not STRENGTH_MIN <= number <= STRENGTH_MAX:
        raise MixError(f"the {which} LoRA's strength is outside {STRENGTH_MIN:g} to {STRENGTH_MAX:g}")
    return round(number, 4)


def check(first, first_strength, second, second_strength, folder_label, name, *,
          networks=None, roots: Iterable[str] | None = None) -> Request:
    """Everything that can be refused before anything is read or written."""
    first, second = str(first or "").strip(), str(second or "").strip()
    if not first or not second:
        raise MixError("choose two LoRAs to mix")
    networks = networks if networks is not None else forge_networks()
    first_path, second_path = lora_file(first, networks), lora_file(second, networks)
    for label, path in ((first, first_path), (second, second_path)):
        if path is None or not path.is_file():
            raise MixError(f"{label} is not a LoRA Forge knows on disk; refresh the list")
        if path.suffix.lower() != ".safetensors":
            raise MixError(f"{label} is not a safetensors file, the only kind this mixer reads")
    if first_path.resolve() == second_path.resolve():
        raise MixError("choose two different LoRAs")
    first_strength = _strength(first_strength, "first")
    second_strength = _strength(second_strength, "second")
    if first_strength == 0 and second_strength == 0:
        raise MixError("both strengths are 0: the mix would change nothing")
    label = str(folder_label or "").strip()
    if not label:
        raise MixError("choose the folder the new LoRA goes in")
    folder = folder_for(label, roots)
    if folder is None:
        raise MixError("that folder is not one of the LoRA folders any more; refresh the list")
    final = check_name(name, first_path.stem, second_path.stem)
    target = folder / f"{final}.safetensors"
    if target.exists() or any(entry.name.casefold() == target.name.casefold() for entry in folder.iterdir()):
        raise MixError(f"{target.name} is already in that folder; choose another name")
    if known_name(final, networks):
        raise MixError(f"Forge already has a LoRA called {final}; choose another name, or its tag would name both")
    return Request(first, first_path, first_strength, second, second_path, second_strength, label, folder, final)


def _link_into_place(temp: Path, target: Path) -> None:
    """The finished file under its name, never over another: a hard link fails
    where a file of that name appeared meanwhile."""
    try:
        os.link(temp, target)
    except FileExistsError:
        raise MixError(f"{target.name} appeared in that folder meanwhile; choose another name") from None
    except OSError:
        if target.exists():
            raise MixError(f"{target.name} appeared in that folder meanwhile; choose another name") from None
        os.replace(temp, target)
        return
    os.unlink(temp)


def run(request: Request, *, progress: Callable[[str], None] = lambda _m: None,
        cancelled: Callable[[], bool] = lambda: False, networks=None) -> dict:
    """Make the mix: read, mix, write, remember, draw, tell Forge. The entry
    remembered, with ``layers`` and ``size`` for the summary."""
    import mc_lora_hashing

    progress(f"Reading {request.first}")
    first_meta, first_tensors = read_safetensors(request.first_path)
    progress(f"Reading {request.second}")
    second_meta, second_tensors = read_safetensors(request.second_path)
    first = Source(request.first, first_tensors, first_meta, request.first_strength)
    second = Source(request.second, second_tensors, second_meta, request.second_strength)

    progress("Mixing")
    outcome = mix(first, second, cancelled=cancelled)
    if cancelled():
        raise MixError("cancelled")

    made = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    inputs = []
    for source, path in ((first, request.first_path), (second, request.second_path)):
        sha = ""
        try:
            sha = mc_lora_hashing.sha256_file(path) or ""
        except Exception:
            sha = ""
        inputs.append({"name": source.label, "file": path.name, "sha256": sha, "strength": source.strength})
    record = {"made_by": MADE_BY, "format": RECORD_FORMAT, "made": made, "method": "ranks concatenated",
              "inputs": inputs}
    metadata = {**carried(first_meta, second_meta), "ss_output_name": request.name,
                "modelspec.title": request.name, METADATA_KEY: json.dumps(record, ensure_ascii=False)}

    progress(f"Writing {request.name}")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=request.folder, prefix=".mc-lora-mix-", suffix=".part",
                                         delete=False) as handle:
            temp = Path(handle.name)
        sha = write_safetensors(temp, metadata, outcome.tensors)
        if cancelled():
            raise MixError("cancelled")
        _link_into_place(temp, request.target)
        temp = None
    except OSError as exc:
        raise MixError(f"the new LoRA could not be written ({exc.strerror or type(exc).__name__})") from None
    finally:
        if temp is not None:
            try:
                temp.unlink()
            except OSError:
                pass

    labels = [(source.label, source.strength) for source in (first, second)]
    placeholder = write_placeholder(request.target, request.name, labels)
    described = write_user_metadata(request.target, [(request.first_path, first.label, first.strength),
                                                     (request.second_path, second.label, second.strength)])
    entry = {"name": request.name, "file": request.target.name, "folder": request.folder_label,
             "path": str(request.target), "sha256": sha, "made": made, "inputs": inputs,
             "rank": outcome.rank, "size": request.target.stat().st_size,
             "layers": {"shared": outcome.shared, "first_only": outcome.only_first,
                        "second_only": outcome.only_second},
             "placeholder": placeholder, "description_written": described}
    remember(entry)
    refresh_forge(request.name, networks)
    sweep_placeholders()
    log(f"made a LoRA mix: {outcome.shared + outcome.only_first + outcome.only_second} layer(s), "
        f"{outcome.shared} changed by both, rank up to {outcome.rank}")
    return entry


def submit(first, first_strength, second, second_strength, folder_label, name):
    """Check the request at once and queue the mix on the LoRA jobs' runner.
    Raises :class:`MixError` for a refusal."""
    import mc_lora_jobs

    request = check(first, first_strength, second, second_strength, folder_label, name)

    def target(job) -> str:
        job.log_line(f"{request.first} × {request.first_strength:g} + {request.second} × "
                     f"{request.second_strength:g} → {request.folder_label}/{request.name}")
        # A refusal is the job's answer, not its failure: the runner writes a
        # failed job's traceback to the log, and a refusal names the LoRAs and
        # the name chosen.
        try:
            entry = run(request, progress=job.set_message, cancelled=lambda: job.cancelled)
        except MixError as exc:
            if job.cancelled:
                return "Cancelled; nothing was written"
            job.log_line(f"Not made: {exc}")
            return f"Not made: {exc}"
        layers = entry["layers"]
        total = layers["shared"] + layers["first_only"] + layers["second_only"]
        picture = "a placeholder picture" if entry["placeholder"] else "your picture kept"
        return (f"Saved {request.name} in {request.folder_label}: {total} layer(s), {layers['shared']} changed by "
                f"both, rank up to {entry['rank']}, {entry['size'] / (1024 * 1024):.1f} MB, {picture}")

    return mc_lora_jobs.MANAGER.submit("mix", f"Mix · {request.name}", target)
