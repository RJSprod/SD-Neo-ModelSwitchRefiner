"""Warm LoRA, Cold LoRA: where a merged LoRA's un-merged originals live.

Forge merges a LoRA *into* the weights rather than holding it beside them, and
it keeps a way back: ``patch_weight_to_device`` files each patched layer's
original in the patcher's ``backup`` on the offload device -- system RAM -- and
``unpatch_model`` writes those originals back whenever the LoRA set changes or
the model is unloaded. That is what makes a LoRA change a round trip through RAM
instead of a read of the checkpoint. It is also RAM the machine does not get
back for as long as the LoRA is merged: on the user's machine free RAM fell from
38.6 GB to 26.5 GB at the first "LoRA state ready" of every session, the whole
12.6 GB transformer, because the LoRA touched nearly every layer.

Two modes, chosen from the Forge Assistant's ⋯ menu or from Settings:

**Warm** is Forge's way, unchanged. The originals stay, a LoRA change costs a
round trip (about twenty seconds in the user's log), the checkpoint is never
read again.

**Cold** frees the originals -- at once when the mode is chosen and nothing is
generating, and after every generation otherwise -- and the loaded model becomes
a *blob* this extension owns: the merged weights are the model now, with no
record of what they were. A blob serves the LoRA set baked into it as it is.
Any other set, including none, *evicts* it: the host's own Unload, so the next
generation reads the checkpoint from disk and merges the new set into fresh
weights. Nothing is ever merged on top of a blob. That rule has two guards: the
prompt is read before the host's LoRA pass (``before_pass``), and the host's own
record is checked again at the last hook before sampling (``before_sampling``),
where a rebuilt LoRA state over a blob is taken away and the generation stopped
rather than let through with the wrong weights.

Turning Warm back on frees nothing and restores nothing: the originals are
gone. The blob stays a blob until its next LoRA change, and from that reload on
the originals are kept, because Warm is on. So the scenario this is built for
reads: RAM is getting low, flip to Cold, twelve gigabytes come back; flip to
Warm later and the *next* LoRA stays warm.

What is never baked: a model only partly on the card (the host applies the LoRA
at run time to the offloaded layers from the very patches a bake would clear),
a model whose LoRA is applied on the fly (there is nothing merged), and a merge
the host has not finished (the stamp on the model does not match the patcher).
Each is said on the console; none is an error.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import mc_lora
import mc_memory

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

OPT_LORA_RAM = "model_chain_lora_ram"
"""The Settings key. Stored as the label, the way every radio here is stored."""

WARM = "Warm"
COLD = "Cold"
MODES = (WARM, COLD)
"""The two labels, and the two words the API uses in lower case."""

BLOB_ATTRIBUTE = "mc_lora_blob"
"""Where a baked model carries its :class:`Blob`. On the ``sd_model`` object,
so it travels with the model through this extension's RAM cache and dies with
it when the host drops the model."""

_GB = 1024 ** 3
LOCK_WAIT_SECONDS = 8.0
"""How long a flip from the menu waits for a warm-up that is moving weights
before answering "later" instead."""

_pending_bake = False
"""Cold was chosen while something was generating or moving weights: bake at
the first safe moment instead."""

_drop_pending = ""
"""The host rebuilt a LoRA state over a blob and the generation was stopped:
the reason, kept so the next generation drops the model before anything else."""

_guard = threading.Lock()


@dataclass(frozen=True)
class Blob:
    """What a baked model remembers about itself."""

    hash: str
    """The host's ``current_lora_hash`` for the set that is merged in."""
    composite: str
    """:func:`mc_lora.composite` of the prompt that produced it -- what the next
    prompt is compared against, before the host's LoRA pass can run."""
    freed_bytes: int
    """The originals' size, as freed. For the console and the menu."""
    baked_at: float


# --------------------------------------------------------------------------- #
# The mode
# --------------------------------------------------------------------------- #


def mode() -> str:
    """``WARM`` or ``COLD``. Anything stored that is neither is Warm: Forge's way,
    and the way that frees nothing by surprise."""
    value = mc_memory.option(OPT_LORA_RAM, WARM)
    return COLD if str(value or "").strip().lower() == COLD.lower() else WARM


def is_cold() -> bool:
    return mode() == COLD


def normalise(value) -> str | None:
    """A label or an API word to one of :data:`MODES`, or None."""
    text = str(value or "").strip().lower()
    for label in MODES:
        if text == label.lower():
            return label
    return None


def set_mode(value) -> dict:
    """Store the mode and act on it now. Returns :func:`status` plus a message.

    Cold frees the originals immediately when nothing is generating or moving
    weights; otherwise the bake is left pending and happens when the generation
    ends. Warm frees and restores nothing: a blob stays a blob until its next
    LoRA change, and the message says so.
    """
    global _pending_bake

    chosen = normalise(value)
    if chosen is None:
        raise ValueError(f"not a LoRA mode: {value!r}")

    _store(chosen)
    message = ""
    if chosen == COLD:
        outcome, detail = bake_now("Cold LoRA chosen")
        if outcome == "baked":
            message = (f"Cold LoRA: {detail} of un-merged weights freed from system RAM. The "
                       "loaded model keeps its LoRA baked in; a LoRA change reloads the "
                       "checkpoint from disk.")
        elif outcome == "busy":
            with _guard:
                _pending_bake = True
            message = ("Cold LoRA: a generation is running; the un-merged weights are freed "
                       "when it ends.")
        elif outcome == "blob":
            message = "Cold LoRA: the loaded model already carries its LoRA baked in."
        elif outcome == "nothing":
            message = ("Cold LoRA: no LoRA is merged into the loaded model, so there is "
                       "nothing to free; the next merge is baked after its generation.")
        else:
            message = f"Cold LoRA: nothing freed — {detail}."
    else:
        with _guard:
            _pending_bake = False
        message = "Warm LoRA: the next LoRA merge keeps its un-merged weights in system RAM."
        if blob_of(_loaded()) is not None:
            message += " The loaded model stays baked until its LoRA changes."
    found = status()
    found["message"] = message
    return found


def _store(label: str) -> None:
    """Write the option the way the Settings page would, and save it."""
    try:
        from modules import shared
    except Exception:
        return
    try:
        changed = shared.opts.set(OPT_LORA_RAM, label)
    except Exception:
        changed = True
        try:
            shared.opts.data[OPT_LORA_RAM] = label
        except Exception:
            logger.debug("Model Chain: could not store the LoRA mode", exc_info=True)
            return
    if changed is False:
        return
    try:
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: the LoRA mode was set but could not be saved", exc_info=True)


# --------------------------------------------------------------------------- #
# Reading the loaded model
# --------------------------------------------------------------------------- #


def _loaded():
    try:
        model = mc_memory.model_data_sd_model()
    except Exception:
        return None
    return model if mc_memory._is_real_model(model) else None


def blob_of(model) -> Blob | None:
    """The :class:`Blob` a model carries, or None for a model with its originals."""
    if model is None:
        return None
    try:
        found = getattr(model, BLOB_ATTRIBUTE, None)
    except Exception:
        return None
    return found if isinstance(found, Blob) else None


def is_blob(model) -> bool:
    return blob_of(model) is not None


_HOLDERS = ("forge_objects", "forge_objects_after_applying_lora", "forge_objects_original")
"""Every place the host keeps a patcher of the loaded model.

``forge_objects`` is reset at the start of every batch from
``forge_objects_original`` and then set from ``forge_objects_after_applying_lora``,
the clone the LoRA loader built -- and another extension can leave a clone of
its own in it by the end of a generation. A bake that cleared the patches of
``forge_objects`` alone would leave the loader's clone holding the LoRA, and the
next generation would merge it onto the merged weights. All three are read, so
the patches go wherever they are.
"""


def _patchers(model) -> list:
    if model is None or not mc_memory._is_real_model(model):
        return []
    found: list = []
    seen: set[int] = set()
    for holder in _HOLDERS:
        objects = getattr(model, holder, None)
        if objects is None:
            continue
        for attr in ("unet", "clip", "vae"):
            patcher = getattr(objects, attr, None)
            if patcher is None:
                continue
            patcher = getattr(patcher, "patcher", patcher)
            if id(patcher) in seen:
                continue
            seen.add(id(patcher))
            found.append(patcher)
    return found


def _tensor_bytes(entry) -> int:
    """Bytes of one ``backup`` entry: a tensor, or Forge Neo's ``(weight, inplace_update)``."""
    weight = getattr(entry, "weight", entry)
    try:
        return int(weight.numel()) * int(weight.element_size())
    except Exception:
        return 0


def originals_bytes(model) -> int:
    """How much RAM the un-merged originals of the loaded LoRA hold. 0 for a blob."""
    total = 0
    for patcher in _patchers(model):
        backup = getattr(patcher, "backup", None)
        patches = getattr(patcher, "patches", None)
        if not isinstance(backup, dict) or not isinstance(patches, dict):
            continue
        try:
            keys = [key for key in list(patches.keys()) if key in backup]
            total += sum(_tensor_bytes(backup[key]) for key in keys if key in backup)
        except Exception:
            continue
    return total


def _unbakeable(patcher) -> str:
    """Why this patcher's merge cannot be baked, or "" when it can."""
    patches = getattr(patcher, "patches", None)
    if not isinstance(patches, dict) or not patches:
        return ""
    inner = getattr(patcher, "model", None)
    try:
        online = patcher.has_online_lora() if callable(getattr(patcher, "has_online_lora", None)) else False
    except Exception:
        online = False
    if online:
        return "its LoRA is applied on the fly, so nothing is merged"
    if getattr(inner, "model_lowvram", False) or int(getattr(inner, "lowvram_patch_counter", 0) or 0) > 0:
        return ("it is only partly on the card, and the host applies the LoRA to the rest "
                "at run time from the very patches a bake would clear")
    stamped = getattr(inner, "current_weight_patches_uuid", None)
    wanted = getattr(patcher, "patches_uuid", None)
    if stamped is None or wanted is None or stamped != wanted:
        return "the host has not finished merging it"
    return ""


# --------------------------------------------------------------------------- #
# Baking
# --------------------------------------------------------------------------- #


def _bake(model, composite: str, reason: str) -> tuple[str, str]:
    """Make the loaded model a blob. Caller holds every lock that matters.

    Returns ``("baked", "12.1 GB")``, ``("blob", "")`` when it already was one,
    ``("nothing", "")`` when no LoRA is merged, or ``("kept", why)``.
    """
    if model is None:
        return "nothing", ""
    if blob_of(model) is not None:
        return "blob", ""
    live = mc_lora.state_of(model)
    patchers = [p for p in _patchers(model) if isinstance(getattr(p, "patches", None), dict)
                and getattr(p, "patches")]
    if live is None or not patchers:
        return "nothing", ""
    for patcher in patchers:
        why = _unbakeable(patcher)
        if why:
            logger.info("Model Chain: Cold LoRA left %s's originals where they are — %s",
                        mc_memory._model_name(patcher), why)
            return "kept", why

    before = mc_memory.free_ram_bytes()
    freed = 0
    for patcher in patchers:
        patches = patcher.patches
        backup = getattr(patcher, "backup", None)
        keys = list(patches.keys())
        if isinstance(backup, dict):
            for key in keys:
                entry = backup.pop(key, None)
                if entry is not None:
                    freed += _tensor_bytes(entry)
        patches.clear()
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    setattr(model, BLOB_ATTRIBUTE, Blob(hash=live, composite=composite, freed_bytes=freed,
                                        baked_at=time.time()))
    after = mc_memory.free_ram_bytes()
    logger.info(
        "Model Chain: Cold LoRA — %s: the un-merged weights of the loaded LoRA (%.1f GB) left "
        "system RAM (%.1f GB -> %.1f GB free). The merged weights are the model now; a change "
        "of LoRA reloads the checkpoint from disk, and nothing is ever merged on top of it",
        reason, freed / _GB, before / _GB, after / _GB)
    return "baked", f"{freed / _GB:.1f} GB"


def bake_now(reason: str) -> tuple[str, str]:
    """Bake the loaded model from outside a generation, if nothing is in the way.

    Takes the host's queue lock only when it is free -- every GPU call the host
    makes runs under it, so holding it is what "nothing is generating" means --
    and this module's model lock for the warm-up that moves weights on its own
    thread. Answers ``("busy", "")`` rather than waiting on either.
    """
    import mc_broker

    if mc_broker.host_busy():
        return "busy", ""
    lock = _host_queue_lock()
    taken = False
    if lock is not None:
        try:
            taken = bool(lock.acquire(blocking=False))
        except TypeError:
            taken = bool(lock.acquire(False))
        except Exception:
            taken = False
        if not taken:
            return "busy", ""
    try:
        if not mc_memory._model_lock.acquire(timeout=LOCK_WAIT_SECONDS):
            return "busy", ""
        try:
            model = _loaded()
            return _bake(model, _composite_of_live(model), reason)
        finally:
            mc_memory._model_lock.release()
    finally:
        if taken:
            try:
                lock.release()
            except Exception:
                pass


def _host_queue_lock():
    try:
        from modules import call_queue

        return getattr(call_queue, "queue_lock", None)
    except Exception:
        return None


def _composite_of_live(model) -> str:
    """The composite a model baked outside a generation is compared against.

    Nothing typed is in hand, so the last prompt this module saw for this model
    is used; failing that the host's hash, which no prompt text can equal, so
    the first generation after such a bake compares unequal and reloads --
    the conservative direction, and said on the console when it happens.
    """
    found = getattr(model, _LAST_COMPOSITE, None)
    if isinstance(found, str):
        return found
    return mc_lora.state_of(model) or ""


_LAST_COMPOSITE = "mc_lora_last_composite"
"""The composite of the last pass that ran on a model, left on the model so a
bake from the menu knows what its LoRA set was asked for as."""


# --------------------------------------------------------------------------- #
# What a pass asks for
# --------------------------------------------------------------------------- #


def _texts(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item or "") for item in value]
    return [str(value)]


def _styled(texts: list[str], styles) -> list[str]:
    """The prompts with their styles applied, as ``setup_prompts`` applies them."""
    if not styles:
        return texts
    try:
        from modules import shared

        apply = getattr(getattr(shared, "prompt_styles", None), "apply_styles_to_prompt", None)
        if not callable(apply):
            return texts
        return [str(apply(text, list(styles))) for text in texts]
    except Exception:
        return texts


def _additional_network() -> str:
    """The tag Settings' *Add network to prompt* adds to every prompt, or ""."""
    try:
        from modules import shared

        name = str(getattr(shared.opts, "sd_lora", "None") or "None")
        if name in ("None", ""):
            return ""
        weight = getattr(shared.opts, "extra_networks_default_multiplier", 1.0)
        return f"<lora:{name}:{weight}>"
    except Exception:
        return ""


def _composite(text: str) -> str:
    """:func:`mc_lora.composite` of one positive prompt, plus the additional
    network when the prompt does not name it -- as the host's LoRA pass adds it."""
    texts = [text]
    additional = _additional_network()
    if additional:
        wanted = additional.split(":")[1].casefold()
        named = any(match.group(1).casefold() == "lora" and
                    match.group(2).split(":")[0].strip().casefold() == wanted
                    for match in mc_lora.RE_EXTRA_NET.finditer(text))
        if not named:
            texts.append(additional)
    return mc_lora.composite(*texts)


def _composites(texts: list[str]) -> list[str]:
    """One composite per prompt, distinct, in order. A batch asks for the first
    prompt's set (``extra_networks.parse_prompts`` keeps only that one), but a
    batch whose prompts disagree is compared whole: the conservative reading."""
    found: list[str] = []
    for text in texts:
        composite = _composite(text)
        if composite not in found:
            found.append(composite)
    return found or [""]


def pass_sets(p) -> list[list[str]]:
    """The LoRA sets this generation will ask the host for, pass by pass.

    The positive prompts only: that is what the host's ``parse_extra_network_prompts``
    reads. With hires fix on, the hires prompts are a second pass when their
    sets differ (an empty hires prompt is the first prompt again, as the host
    reads it).
    """
    styles = getattr(p, "styles", None) or []
    base = _composites(_styled(_texts(getattr(p, "prompt", "")), styles))
    found = [base]
    if getattr(p, "enable_hr", False):
        hr = _texts(getattr(p, "hr_prompt", ""))
        if any(text.strip() for text in hr):
            second = _composites(_styled(hr, styles))
            if second != base:
                found.append(second)
    return found


def pass_composites(p) -> list[str]:
    """Every distinct composite the generation asks for, across its passes."""
    found: list[str] = []
    for part in pass_sets(p):
        for composite in part:
            if composite not in found:
                found.append(composite)
    return found


def bake_composite(p) -> str:
    """What the host will have merged when the generation ends: the first
    prompt's set of the last pass, which is the one the host applies."""
    return pass_sets(p)[-1][0]


def describe(composite: str) -> str:
    if not composite:
        return "no LoRA"
    tags = [line for line in composite.split("\n") if line]
    return ", ".join(tags) if len(tags) <= 3 else f"{len(tags)} extra-network tags"


# --------------------------------------------------------------------------- #
# The hooks
# --------------------------------------------------------------------------- #


def before_pass(p) -> str:
    """Before the host's LoRA pass: evict a blob the prompt would change.

    Called from ``before_process`` -- the host reloads a dropped model from disk
    right after it, in ``manage_model_and_prompt_cache``, and only then parses
    the prompt's networks and merges them into fresh weights. Also the moment a
    bake left pending by a busy flip is made, before the comparison, since a
    model baked here is compared against this very prompt.

    Returns the reason the model was dropped, or "".
    """
    global _drop_pending

    model = _loaded()
    if model is None:
        with _guard:
            _drop_pending = ""
        return ""
    try:
        sets = pass_sets(p)
    except Exception:
        logger.debug("Model Chain: could not read the prompt's LoRA set", exc_info=True)
        sets = []
    if sets:
        try:
            setattr(model, _LAST_COMPOSITE, sets[-1][0])
        except Exception:
            pass

    with _guard:
        pending = _pending_bake
        stopped = _drop_pending
        _drop_pending = ""
    if pending and sets and blob_of(model) is None:
        _take_pending(model, sets[-1][0], "Cold LoRA chosen during a generation")

    blob = blob_of(model)
    if blob is None:
        return ""
    if stopped:
        reason = stopped
    elif sets and all(part == blob.composite for group in sets for part in group):
        return ""
    else:
        if not sets:
            asked = "an unreadable prompt"
            where = "the prompt"
        elif all(part == blob.composite for part in sets[0]):
            # The first pass runs on the blob and the hires pass would merge
            # another set over it half way through the generation.
            asked = describe(next(part for part in sets[1] if part != blob.composite))
            where = "the hires pass"
        else:
            asked = describe(next(part for part in sets[0] if part != blob.composite))
            where = "the prompt"
        reason = f"{where} asks for {asked} and the model carries {describe(blob.composite)} baked in"
    parked = mc_memory.drop_image_model(f"Cold LoRA — {reason}")
    if parked:
        logger.info("Model Chain: Cold LoRA — %s; the checkpoint reloads from disk for this "
                    "generation rather than merging on top of a blob", reason)
    else:
        logger.warning("Model Chain: Cold LoRA — %s, and the model could not be dropped; the "
                       "host's LoRA state is being checked again before sampling", reason)
    return reason


def _take_pending(model, composite: str, reason: str) -> None:
    global _pending_bake

    with mc_memory._model_lock:
        outcome, _ = _bake(model, composite, reason)
    if outcome in ("baked", "blob", "nothing"):
        with _guard:
            _pending_bake = False


def before_sampling(p) -> str:
    """The last hook before the sampler: a LoRA state rebuilt over a blob is undone.

    The prompt was read before the host's pass, so this is the net under it:
    anything that reaches the host's hash without reaching the prompt text --
    a LoRA file renamed, a setting changed between the two -- lands here. The
    patches are taken off the clones before any weight moves, the host's record
    is put back to what is really merged, the generation is interrupted rather
    than finished on the wrong weights, and the model is dropped at the start
    of the next one. Returns the reason, or "".
    """
    global _drop_pending

    model = _loaded()
    blob = blob_of(model)
    if blob is None:
        return ""
    live = mc_lora.state_of(model)
    if live == blob.hash:
        # The host's record still names the baked set, so its loader took its
        # early return. Weight patches another extension adds for one pass are
        # its own business: the host files the blob's weight as their original
        # and puts it back afterwards, which is not a LoRA on top of a LoRA.
        return ""
    patched = [patcher for patcher in _patchers(model)
               if isinstance(getattr(patcher, "patches", None), dict) and patcher.patches]
    for patcher in patched:
        try:
            inner = getattr(patcher, "model", None)
            stamped = getattr(inner, "current_weight_patches_uuid", None)
            patcher.patches.clear()
            if stamped is not None:
                patcher.patches_uuid = stamped
        except Exception:
            logger.debug("Model Chain: could not take a rebuilt LoRA state off a blob",
                         exc_info=True)
    try:
        setattr(model, mc_lora.HASH_ATTRIBUTE, blob.hash)
    except Exception:
        pass
    reason = ("the host rebuilt its LoRA state over a model that carries "
              f"{describe(blob.composite)} baked in")
    with _guard:
        _drop_pending = reason
    logger.warning(
        "Model Chain: Cold LoRA — %s. The new patches were taken away before any weight "
        "moved, this generation is stopped rather than finished on the wrong weights, and "
        "the checkpoint reloads from disk at the next one. Press Generate again", reason)
    _stop(p, reason)
    return reason


def _stop(p, reason: str) -> None:
    try:
        from modules import shared

        state = shared.state
        if callable(getattr(state, "interrupt", None)):
            state.interrupt()
        else:
            state.interrupted = True
    except Exception:
        pass
    try:
        comment = getattr(p, "comment", None)
        if callable(comment):
            comment(f"Model Chain: stopped — {reason}; press Generate again")
    except Exception:
        pass


def after_generation(p=None, texts=None) -> str:
    """After a pass: in Cold, bake what the host merged. Returns the outcome word.

    ``texts`` are the positive prompts of the pass that ran when ``p`` is not
    the processing object that describes it -- the Stage 2 refine, whose prompt
    is not Stage 1's.
    """
    global _pending_bake

    with _guard:
        pending = _pending_bake
    if not pending and not is_cold():
        return "warm"
    model = _loaded()
    if model is None:
        return "nothing"
    try:
        if texts is not None:
            composite = _composites(_texts(texts))[0]
        elif p is not None:
            composite = bake_composite(p)
        else:
            composite = _composite_of_live(model)
    except Exception:
        logger.debug("Model Chain: could not read the pass's LoRA set", exc_info=True)
        composite = _composite_of_live(model)
    try:
        setattr(model, _LAST_COMPOSITE, composite)
    except Exception:
        pass
    with mc_memory._model_lock:
        outcome, _ = _bake(model, composite, "after the generation")
    if outcome in ("baked", "blob", "nothing"):
        with _guard:
            _pending_bake = False
    return outcome


# --------------------------------------------------------------------------- #
# For the menu
# --------------------------------------------------------------------------- #


def status() -> dict:
    """What the ⋯ menu shows: the mode, and what the loaded model carries."""
    model = _loaded()
    blob = blob_of(model)
    merged = model is not None and blob is None and mc_lora.state_of(model) is not None
    with _guard:
        pending = _pending_bake
    name = ""
    try:
        info = getattr(model, "sd_checkpoint_info", None)
        name = str(getattr(info, "name_for_extra", "") or "") if info is not None else ""
    except Exception:
        name = ""
    return {
        "mode": mode().lower(),
        "merged": bool(merged),
        "blob": blob is not None,
        "originals_bytes": originals_bytes(model) if merged else 0,
        "freed_bytes": int(blob.freed_bytes) if blob is not None else 0,
        "pending": bool(pending),
        "model": name,
    }


def reset_for_tests() -> None:
    global _pending_bake, _drop_pending

    with _guard:
        _pending_bake = False
        _drop_pending = ""
