"""Model residency and cache management for the Model Chain extension.

Design principle (section 4.1): VRAM-FIRST, DEMOTE ONLY UNDER PRESSURE.
The residency cascade is VRAM -> system RAM -> disk, and it is demand driven
rather than a fixed evict-on-switch rule.


How this maps onto Forge Neo
----------------------------
Forge Neo keeps exactly one checkpoint in ``sd_models.model_data.sd_model``, and
``forge_model_reload()`` unconditionally calls ``memory_management.unload_all_models()``
and drops its reference before loading a new checkpoint from disk. Section 4.4
forbids reimplementing offload logic or moving raw state dicts between devices,
so we do not fight that path -- we cooperate with it:

* ``unload_all_models()`` calls ``ModelPatcher.detach()``, which moves weights to
  the patcher's *offload device* (system RAM). It does not free them. The only
  reason a checkpoint normally disappears from RAM is that ``forge_model_reload()``
  drops the last Python reference to it and the garbage collector reclaims it.

* So keeping a checkpoint resident in system RAM is exactly a matter of holding
  a reference to its ``sd_model`` across the switch. That is what this cache is.

* Restoring a cached checkpoint is then a pointer swap: put the object back into
  ``model_data``, set ``forge_hash`` to match ``forge_loading_parameters``, and the
  next ``forge_model_reload()`` returns early instead of touching the disk. No
  ``unload_all_models()`` runs on that path, so the *incoming* model is brought
  back to VRAM lazily by ``memory_management.load_models_gpu()``, which evicts the
  outgoing one only if the VRAM is actually needed. That is the demand-driven
  cascade section 4.2 asks for, implemented entirely through host entry points.

One consequence is worth stating plainly: the very first load of a checkpoint has
to go through ``forge_model_reload()``, and that host function unloads everything
before it loads. A cold first switch therefore always demotes the outgoing model
to RAM even when both models would have fit in VRAM together. Every subsequent
switch between two cached checkpoints is a warm pointer swap with demand-driven
VRAM eviction, which is where the "both models stay hot in VRAM" behaviour and
the measurable speedup over a disk reload come from.


Forward compatibility (section 4.3)
-----------------------------------
The two-slot limit lives behind ``ensure_resident()`` / ``get_model()`` and is a
private detail of this module. ``model_chain.py`` never assumes a slot count, so
swapping ``_Cache`` for an LRU pool of 3+ models needs no change to the
orchestration code -- only ``_MAX_SLOTS`` and ``_Cache.admit()``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")

_MAX_SLOTS = 2
"""v1 holds Model A and Model B. See the forward-compatibility note above."""

_GB = 1024**3

VRAM_HEADROOM_BYTES = 1 * _GB
"""Spare VRAM kept free for activations when judging whether two models fit."""

RAM_RESERVE_BYTES = 2 * _GB
"""System RAM never handed to the cache, so the host process cannot be OOM-killed."""


class ModelChainError(RuntimeError):
    """Raised when a requested checkpoint cannot be resolved or loaded."""


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


MODEL_FLAGS = ("kontext", "edit", "nunchaku", "klein", "wan", "pid", "anima", "krea2")
"""``backend.args.dynamic_args`` fields that describe *which model is loaded*.

``forge_loader`` sets these every time it loads a checkpoint, but a warm swap
never runs the loader, so they have to be carried with the cached model.
Leaving them describing the previously loaded checkpoint is not cosmetic:
``nunchaku`` selects a different LoRA application path, ``kontext`` and ``edit``
decide whether Flux.1 and Qwen-Image use reference conditioning, ``klein``
changes sampling, and ``pid`` changes the latent shape.
"""


def snapshot_model_flags() -> dict:
    """Capture the loader-set flags describing the currently loaded model."""
    try:
        from backend.args import dynamic_args

        return {name: getattr(dynamic_args, name, False) for name in MODEL_FLAGS}
    except Exception:
        return {}


def _apply_model_flags(flags: dict) -> None:
    """Re-apply captured flags and clear per-generation latent state.

    Mirrors what ``forge_loader`` does after a real load: ``dynamic_args.reset()``
    followed by assigning each model flag.
    """
    if not flags:
        return
    try:
        from backend.args import dynamic_args

        dynamic_args.reset()
        for name, value in flags.items():
            setattr(dynamic_args, name, value)
    except Exception:
        logger.warning("Model Chain: failed to restore model flags on warm swap", exc_info=True)


@dataclass
class _Entry:
    key: str
    """``str(model_data.forge_loading_parameters)`` -- what ``forge_hash`` compares against."""
    checkpoint_name: str
    sd_model: object
    size_bytes: int
    model_flags: dict = field(default_factory=dict)
    """``dynamic_args`` state as of this model's load; see ``MODEL_FLAGS``."""
    last_used: float = field(default_factory=time.monotonic)


class _Cache:
    """Fixed-capacity residency cache keyed by loading parameters."""

    def __init__(self, capacity: int = _MAX_SLOTS):
        self._capacity = capacity
        self._entries: dict[str, _Entry] = {}

    # -- queries ---------------------------------------------------------- #

    def get(self, key: str) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is not None:
            entry.last_used = time.monotonic()
        return entry

    def has(self, key: str) -> bool:
        return key in self._entries

    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def names(self) -> list[str]:
        return [e.checkpoint_name for e in self._entries.values()]

    # -- mutation --------------------------------------------------------- #

    def admit(self, entry: _Entry, budget_bytes: int) -> bool:
        """Take ``entry`` into the cache, evicting as needed to honour the budget.

        Returns False when the entry cannot be cached at all, in which case the
        caller must let the model be released (a disk reload next time). Host
        RAM exhaustion is far more disruptive than a VRAM OOM -- it can hang or
        kill the whole process -- so this check is fail-safe, never best-effort
        (section 4.5).
        """
        if entry.key in self._entries:
            self._entries[entry.key] = entry
            return True

        if entry.size_bytes > budget_bytes:
            logger.info(
                "Model Chain: %s (%.1f GB) exceeds the whole cache budget (%.1f GB); not caching",
                entry.checkpoint_name,
                entry.size_bytes / _GB,
                budget_bytes / _GB,
            )
            return False

        while self._entries and (
            len(self._entries) >= self._capacity
            or self.total_bytes() + entry.size_bytes > budget_bytes
        ):
            self._evict_oldest()

        if self.total_bytes() + entry.size_bytes > budget_bytes:
            return False

        self._entries[entry.key] = entry
        return True

    def _evict_oldest(self) -> None:
        oldest = min(self._entries.values(), key=lambda e: e.last_used)
        self.drop(oldest.key)

    def drop(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        logger.info("Model Chain: releasing cached model %s", entry.checkpoint_name)
        entry.sd_model = None

    def clear(self) -> None:
        for key in list(self._entries):
            self.drop(key)


_cache = _Cache()

_pending_restore: str | None = None
"""Checkpoint the extension owes a switch back to, set when Stage 2 leaves
Model B loaded while the UI selection has been put back to Model A."""


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #


def _loading_parameters_key() -> str:
    from modules.sd_models import model_data

    return str(model_data.forge_loading_parameters)


def _is_real_model(model) -> bool:
    """False for ``FakeInitialModel``, the placeholder used before any load."""
    if model is None:
        return False
    return type(model).__name__ != "FakeInitialModel"


def checkpoint_info(name: str):
    from modules import sd_models

    if not name or name in ("None", "none"):
        return None
    return sd_models.get_closet_checkpoint_match(name)


def file_size_bytes(name: str) -> int:
    """Disk footprint of a checkpoint plus the additional modules loaded with it.

    Used to size a model that is not currently loaded. The VAE and text encoders
    of Flux-family models live in separate files selected as additional modules,
    so a checkpoint file size alone would badly under-count them.
    """
    from modules import shared

    total = 0
    info = checkpoint_info(name)
    if info is not None:
        try:
            total += os.path.getsize(info.filename)
        except OSError:
            pass

    for module in getattr(shared.opts, "forge_additional_modules", []) or []:
        try:
            total += os.path.getsize(module)
        except OSError:
            pass

    return total


def loaded_size_bytes(sd_model) -> int:
    """Actual footprint of a loaded model, summed across its patchers."""
    if not _is_real_model(sd_model):
        return 0

    objects = getattr(sd_model, "forge_objects", None)
    if objects is None:
        return 0

    total = 0
    seen: set[int] = set()
    for attr in ("unet", "clip", "vae"):
        patcher = getattr(objects, attr, None)
        if patcher is None:
            continue
        patcher = getattr(patcher, "patcher", patcher)
        if id(patcher) in seen:
            continue
        seen.add(id(patcher))
        try:
            total += int(patcher.model_size())
        except Exception:
            continue
    return total


def free_vram_bytes() -> int:
    try:
        from backend import memory_management

        return int(memory_management.get_free_memory(memory_management.get_torch_device()))
    except Exception:
        return 0


def free_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def total_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def cache_budget_bytes() -> int:
    """Configured ceiling for RAM held by the cache (section 4.5)."""
    from modules import shared

    configured = getattr(shared.opts, "model_chain_ram_budget_gb", 0) or 0
    if configured > 0:
        return int(configured * _GB)

    total = total_ram_bytes()
    if total <= 0:
        # System RAM could not be queried. Refusing to cache is the fail-safe
        # answer: models reload from disk, which is slow but cannot OOM the
        # host. Set the budget explicitly in Settings to override this.
        logger.warning(
            "Model Chain: unable to detect system RAM; the model cache is disabled. "
            'Set "Model Chain: max system RAM for model cache (GB)" in Settings to enable it.'
        )
        return 0

    # Conservative default: a third of detected system RAM.
    return int(total / 3)


def default_ram_budget_gb() -> float:
    total = total_ram_bytes()
    if total <= 0:
        return 16.0
    return round(total / _GB / 3, 1)


# --------------------------------------------------------------------------- #
# Residency planning (section 6.6)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResidencyPlan:
    kind: str
    """One of ``warm``, ``dual``, ``offload``, ``disk`` or ``unavailable``."""
    message: str

    @property
    def is_warning(self) -> bool:
        return self.kind in ("disk", "unavailable")


def plan(target_name: str) -> ResidencyPlan:
    """Predict what the switch to ``target_name`` will cost, for the UI status line."""
    if not target_name or target_name in ("None", "none"):
        return ResidencyPlan("unavailable", "No Stage 2 model selected.")

    info = checkpoint_info(target_name)
    if info is None:
        return ResidencyPlan("unavailable", f'Checkpoint "{target_name}" was not found.')

    from modules import shared

    current = shared.sd_model
    current_size = loaded_size_bytes(current)
    target_size = file_size_bytes(target_name)

    if _cache.has(_target_key_for(target_name)):
        return ResidencyPlan(
            "warm",
            f"{info.name_for_extra} is cached in RAM — switch is a warm swap, no disk read.",
        )

    free_vram = free_vram_bytes()
    if free_vram <= 0:
        return ResidencyPlan(
            "unavailable",
            "Unable to query free VRAM — residency cannot be predicted.",
        )

    if free_vram >= target_size + VRAM_HEADROOM_BYTES:
        return ResidencyPlan(
            "dual",
            f"Both models fit in VRAM — no offload expected "
            f"({free_vram / _GB:.1f} GB free, {target_size / _GB:.1f} GB needed).",
        )

    budget = cache_budget_bytes()
    ram_headroom = min(free_ram_bytes() - RAM_RESERVE_BYTES, budget)
    if current_size and ram_headroom >= current_size:
        return ResidencyPlan(
            "offload",
            f"Insufficient VRAM for both — Model A will offload to system RAM "
            f"({free_vram / _GB:.1f} GB VRAM free, {target_size / _GB:.1f} GB needed).",
        )

    return ResidencyPlan(
        "disk",
        f"Insufficient RAM for cache — Model A will reload from disk on switch "
        f"(needs {current_size / _GB:.1f} GB, {max(ram_headroom, 0) / _GB:.1f} GB available to the cache).",
    )


def _target_key_for(name: str) -> str:
    """Best-effort cache key for a checkpoint that is not currently selected.

    Mirrors how ``main_entry.refresh_model_loading_parameters`` builds
    ``forge_loading_parameters`` so the prediction matches the real key without
    mutating any global state.
    """
    try:
        from modules import shared
        from modules_forge.main_entry import forge_unet_storage_dtype_options

        info = checkpoint_info(name)
        if info is None:
            return ""
        unet_storage_dtype, _ = forge_unet_storage_dtype_options.get(
            shared.opts.forge_unet_storage_dtype, (None, False)
        )
        return str(
            dict(
                checkpoint_info=info,
                additional_modules=shared.opts.forge_additional_modules,
                unet_storage_dtype=unet_storage_dtype,
            )
        )
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Residency control
# --------------------------------------------------------------------------- #


def _stash_current() -> None:
    """Move the currently loaded model into the cache before it is displaced."""
    from modules.sd_models import model_data

    model = model_data.sd_model
    if not _is_real_model(model):
        return

    key = _loading_parameters_key()
    if _cache.has(key):
        _cache.get(key)  # refresh recency
        return

    info = getattr(model, "sd_checkpoint_info", None)
    name = info.name_for_extra if info is not None else "(unknown)"
    size = loaded_size_bytes(model) or file_size_bytes(name)

    budget = min(cache_budget_bytes(), max(free_ram_bytes() - RAM_RESERVE_BYTES, 0) + size)
    entry = _Entry(
        key=key,
        checkpoint_name=name,
        sd_model=model,
        size_bytes=size,
        model_flags=snapshot_model_flags(),
    )

    if _cache.admit(entry, budget):
        logger.info("Model Chain: holding %s in the RAM cache (%.1f GB)", name, size / _GB)
    else:
        logger.warning(
            "Model Chain: not enough system RAM to cache %s — it will reload from disk next time",
            name,
        )


def ensure_resident(name: str) -> str:
    """Make ``name`` the loaded checkpoint, and report how it got there.

    Returns ``"unchanged"``, ``"warm"`` (restored from the RAM cache) or
    ``"cold"`` (read from disk). This is the only entry point the orchestration
    code uses to change models; it knows nothing about slots.
    """
    from modules import processing, sd_models, shared
    from modules.sd_models import model_data
    from modules_forge import main_entry

    info = checkpoint_info(name)
    if info is None:
        raise ModelChainError(f'Stage 2 checkpoint "{name}" was not found.')

    if _is_real_model(model_data.sd_model):
        current_info = getattr(model_data.sd_model, "sd_checkpoint_info", None)
        if current_info is not None and current_info.filename == info.filename:
            return "unchanged"

    _stash_current()

    # Let the host recompute loading parameters, dynamic LoRA flags and the
    # checkpoint selection. save=False keeps this transient switch out of the
    # user's config.json.
    main_entry.checkpoint_change(name, None, save=False, refresh=True)
    if str(model_data.forge_loading_parameters.get("checkpoint_info", "")) != str(info):
        main_entry.refresh_model_loading_parameters()

    target_key = _loading_parameters_key()
    entry = _cache.get(target_key)

    if entry is not None and _is_real_model(entry.sd_model):
        # Warm path: pointer swap. forge_model_reload() will return early, so no
        # unload_all_models() runs and VRAM is reclaimed only on demand.
        model_data.set_sd_model(entry.sd_model)
        model_data.forge_hash = target_key
        shared.opts.data["sd_checkpoint_hash"] = entry.sd_model.sd_checkpoint_info.sha256
        # The loader did not run, so re-apply the model flags it would have set.
        _apply_model_flags(entry.model_flags)
        processing.need_global_unload = False
        logger.info("Model Chain: restored %s from the RAM cache", entry.checkpoint_name)
        return "warm"

    sd_models.forge_model_reload()
    return "cold"


def restore_selection(name: str) -> None:
    """Point the UI selection and loading parameters back at ``name``.

    Deliberately does *not* reload. Section 3.1 allows exactly one checkpoint
    switch per Generate click, so the swap back to Model A is deferred to the
    start of the next generation, where ``reinstate_pending()`` turns it into a
    warm pointer swap.
    """
    global _pending_restore

    info = checkpoint_info(name)
    if info is None:
        return

    from modules_forge import main_entry

    main_entry.checkpoint_change(name, None, save=False, refresh=True)
    _pending_restore = name


def reinstate_pending() -> bool:
    """Warm-swap the selected checkpoint back in, if we left a different one loaded.

    Called at the top of every generation. Returns True when a swap happened.
    """
    global _pending_restore

    from modules import processing
    from modules.sd_models import model_data

    if _pending_restore is None:
        return False

    key = _loading_parameters_key()
    entry = _cache.get(key)
    if entry is None or not _is_real_model(entry.sd_model):
        # Nothing cached: leave it to forge_model_reload() to load from disk.
        _pending_restore = None
        return False

    if model_data.sd_model is entry.sd_model:
        _pending_restore = None
        return False

    _stash_current()
    model_data.set_sd_model(entry.sd_model)
    model_data.forge_hash = key
    _apply_model_flags(entry.model_flags)
    processing.need_global_unload = False
    logger.info("Model Chain: restored %s from the RAM cache", entry.checkpoint_name)
    _pending_restore = None
    return True


def clear_references() -> None:
    """Drop reference latents held by the loaded model.

    Edit-capable engines build their reference list from the engine's own
    ``ref_latents`` (populated by ImageStitch) plus the img2img input. Stage 2
    runs with scripts disabled, so ImageStitch never gets to refresh or clear
    that list -- and because a cached model keeps its engine object across
    generations, entries from an earlier run would otherwise be silently mixed
    into every refined image.
    """
    try:
        from backend.args import dynamic_args
        from modules import shared

        model = shared.sd_model
        if hasattr(model, "clear_references"):
            model.clear_references()
        if getattr(model, "ini_latent", None) is not None:
            model.ini_latent = None
        dynamic_args.ref_latents.clear()
    except Exception:
        logger.warning("Model Chain: failed to clear stale references", exc_info=True)


def get_model(name: str):
    """Return the cached ``sd_model`` for ``name``, or None if it is not resident."""
    entry = _cache.get(_target_key_for(name))
    return entry.sd_model if entry is not None else None


def cached_names() -> list[str]:
    return _cache.names()


def release_all() -> None:
    """Drop every cached model. Next use of any of them reloads from disk."""
    global _pending_restore

    _pending_restore = None
    _cache.clear()

    try:
        import gc

        from backend import memory_management

        gc.collect()
        memory_management.soft_empty_cache()
    except Exception:
        pass
