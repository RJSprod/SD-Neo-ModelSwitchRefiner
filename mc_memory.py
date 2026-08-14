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
import threading
import time
from dataclasses import dataclass, field

def _make_logger() -> logging.Logger:
    """Console logger using the host's Rich formatting.

    ``logging.getLogger`` alone yields a logger with no handler, so every
    diagnostic this extension emits would be swallowed -- including the cache
    decisions that explain why a switch was slow. ``setup_logger`` attaches the
    same handler the host's own modules use.
    """
    log = logging.getLogger("model_chain")
    try:
        from backend.logging import setup_logger

        setup_logger(log)
    except Exception:
        log.setLevel(logging.INFO)
    return log


logger = _make_logger()

_model_lock = threading.RLock()
"""Serialises this module's own model movement.

Everything here normally runs on the generation thread, one call at a time. The
background Stage 1 preload is the exception, so every entry point that moves or
reassigns weights takes this lock and the preload holds it for its whole run.
Re-entrant because the preload calls several of those entry points itself.
"""

OPT_PIN_ENCODERS = "model_chain_pin_stage1_encoders"
OPT_PRELOAD = "model_chain_preload_stage1"


def option(name: str, default):
    """Read one of this extension's settings, falling back before the UI exists.

    Settings registered through ``on_ui_settings`` are absent from ``opts``
    during early imports and in tests, and a missing setting must not change
    behaviour -- so the default is the documented behaviour, not a disabled one.
    """
    try:
        from modules import shared

        value = getattr(shared.opts, name, None)
    except Exception:
        return default
    return default if value is None else value

_MAX_SLOTS = 2
"""v1 holds Model A and Model B. See the forward-compatibility note above."""

_GB = 1024**3

VRAM_HEADROOM_BYTES = 1 * _GB
"""Baseline spare VRAM kept free for activations when judging whether two models fit."""

VRAM_HEADROOM_PER_MEGAPIXEL = 0.75 * _GB
"""Additional activation allowance per megapixel of output.

A flat allowance badly under-estimates a large pass: attention activations grow
with pixel count, so a 2048x2048 refine needs several times what 1024x1024
does. Getting this wrong is not a graceful failure -- on Windows the driver
silently spills the overflow to system RAM over PCIe, and sampling drops from
sub-second to tens of seconds per step with no error to explain it.
"""

VRAM_MODEL_OVERHEAD_FRACTION = 0.15
"""How much larger a model is in VRAM than its file is on disk.

``file_size_bytes`` measures the checkpoint on disk, but that is a floor, not
the resident footprint: weights stored below the compute dtype are widened on
load, and LoRA patches and the parameter bookkeeping the patcher keeps alongside
them are not in the file at all.

Under-estimating here is what produces the ``Unloaded partially`` lines in the
log immediately before a UNet load. ``free_memory`` hits our target and stops,
the target turns out to be short, and the host then makes up the difference the
expensive way -- unloading in chunks *while* it loads, which is the slow path
this module exists to avoid. Two measured chains were short by 11% and 14% of
model size, so the correction is proportional to the model rather than flat.

Over-estimating is much cheaper than under-estimating: it evicts a model that
might have fitted, and that model is still warm in the RAM cache.
"""


def vram_headroom_bytes(width: int = 0, height: int = 0) -> int:
    """Activation headroom to keep free for a pass at the given size."""
    if width <= 0 or height <= 0:
        return int(VRAM_HEADROOM_BYTES)

    megapixels = (width * height) / 1_000_000
    return int(VRAM_HEADROOM_BYTES + VRAM_HEADROOM_PER_MEGAPIXEL * max(megapixels - 1.0, 0.0))


def vram_required_bytes(name: str, modules=None, width: int = 0, height: int = 0) -> int:
    """VRAM a pass on ``name`` needs: the model, resident, plus its activations."""
    model = file_size_bytes(name, modules) * (1.0 + VRAM_MODEL_OVERHEAD_FRACTION)
    return int(model + vram_headroom_bytes(width, height))

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

_last_refusal: str | None = None
"""Most recent model the cache could not take, if any.

Lets a cold load say whether it was a first load (expected, once) or the
consequence of a refused cache entry (actionable), instead of pointing at the
RAM setting every time.
"""


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #


def _loading_parameters_key() -> str:
    """Cache key for whatever the loading parameters currently *select*."""
    from modules.sd_models import model_data

    return str(model_data.forge_loading_parameters)


def _loaded_model_key() -> str:
    """Cache key for the model that is actually loaded right now.

    Deliberately not ``_loading_parameters_key()``. ``forge_loading_parameters``
    describes the *selection*, which routinely runs ahead of the loaded model:
    ``restore_selection`` points it back at Stage 1 while Stage 2's model is
    still the one in memory. Keying a stash off the selection there would file
    the outgoing model under the incoming model's key -- and since that key is
    already in the cache, the stash would be skipped entirely and the model
    dropped.

    ``forge_hash`` is by definition the loading-parameter string that produced
    the resident model, which is exactly what a later lookup will compare
    against.
    """
    from modules.sd_models import model_data

    return str(getattr(model_data, "forge_hash", "") or "")


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


INHERIT_MODULES = "Use same choices"
"""Sentinel meaning "keep Stage 1's VAE / text encoder selection".

Same spelling the host's own Hires VAE/TE dropdown uses, so the control reads
familiarly and infotext written by either is interchangeable.
"""


def resolve_modules(values) -> list[str] | None:
    """Turn a VAE/TE selection into the list of module paths to load.

    Returns None when Stage 1's selection should be inherited. An empty list is
    meaningful and distinct from None: it means "no additional modules", i.e.
    use whatever VAE and text encoder are built into the checkpoint.

    Resolution mirrors ``main_entry.modules_change``: names are looked up in the
    host's module list and the result is sorted, so a selection compares equal
    to what the host will actually store.
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    if INHERIT_MODULES in values:
        return None

    from modules_forge.main_entry import module_list

    resolved = []
    for value in values:
        name = os.path.basename(str(value))
        if name in module_list:
            resolved.append(module_list[name])
        else:
            logger.warning("Model Chain: VAE/text encoder module %r was not found", value)

    return sorted(resolved)


def current_modules() -> list[str]:
    from modules import shared

    return list(getattr(shared.opts, "forge_additional_modules", []) or [])


def file_size_bytes(name: str, modules=None) -> int:
    """Disk footprint of a checkpoint plus the additional modules loaded with it.

    Used to size a model that is not currently loaded. The VAE and text encoders
    of Flux-family models live in separate files selected as additional modules,
    so a checkpoint file size alone would badly under-count them -- a Flux.2
    text encoder is several GB on its own.

    ``modules`` sizes a specific selection; None falls back to whatever is
    currently selected.
    """
    total = 0
    info = checkpoint_info(name)
    if info is not None:
        try:
            total += os.path.getsize(info.filename)
        except OSError:
            pass

    resolved = resolve_modules(modules)
    if resolved is None:
        resolved = current_modules()

    for module in resolved:
        try:
            total += os.path.getsize(module)
        except OSError:
            pass

    return total


def model_patchers(sd_model, attrs=("unet", "clip", "vae")) -> list:
    """The ``ModelPatcher`` objects behind a loaded checkpoint, in load order.

    ``forge_objects.clip`` wraps its patcher rather than being one, so each
    entry is unwrapped before use. Duplicates are dropped: some architectures
    share one patcher between two slots, and both ``free_memory`` and
    ``load_models_gpu`` would otherwise see it twice.
    """
    if not _is_real_model(sd_model):
        return []

    objects = getattr(sd_model, "forge_objects", None)
    if objects is None:
        return []

    found: list = []
    seen: set[int] = set()
    for attr in attrs:
        patcher = getattr(objects, attr, None)
        if patcher is None:
            continue
        patcher = getattr(patcher, "patcher", patcher)
        if id(patcher) in seen:
            continue
        seen.add(id(patcher))
        found.append(patcher)
    return found


def loaded_size_bytes(sd_model) -> int:
    """Actual footprint of a loaded model, summed across its patchers."""
    total = 0
    for patcher in model_patchers(sd_model):
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


def total_vram_bytes() -> int:
    try:
        from backend import memory_management

        return int(memory_management.get_total_memory(memory_management.get_torch_device()))
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


DEFAULT_BUDGET_FRACTION = 0.6
"""Share of system RAM the cache may hold by default.

This is a *ceiling*, not a reservation. The real guard is the live free-RAM
check in ``_stash_current``, which refuses an entry that would push available
memory below ``RAM_RESERVE_BYTES``.

An earlier default of one third was too tight to be useful: a Flux.2 Klein
checkpoint with a Qwen3 text encoder occupies roughly 14 GB resident, so on a
32 GB machine the budget came to 10.6 GB and the model was refused outright --
every switch then cold-loaded from disk, which is exactly the cost this cache
exists to avoid. Because weights are *moved* between devices rather than
copied, the cache mostly holds whichever model is not currently in VRAM, so a
ceiling near a large model's full size is the useful setting.
"""


def cache_budget_bytes() -> int:
    """Ceiling for RAM held by the cache (section 4.5)."""
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

    return int(total * DEFAULT_BUDGET_FRACTION)


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


def plan(target_name: str, modules=None) -> ResidencyPlan:
    """Predict what the switch to ``target_name`` will cost, for the UI status line."""
    if not target_name or target_name in ("None", "none"):
        return ResidencyPlan("unavailable", "No Stage 2 model selected.")

    info = checkpoint_info(target_name)
    if info is None:
        return ResidencyPlan("unavailable", f'Checkpoint "{target_name}" was not found.')

    from modules import shared

    current = shared.sd_model
    current_size = loaded_size_bytes(current)
    target_size = file_size_bytes(target_name, modules)

    if _cache.has(_target_key_for(target_name, modules)):
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

    # Predict against the same figure make_vram_room() will act on, so the
    # pre-flight label does not promise dual residency the switch then breaks.
    target_required = vram_required_bytes(target_name, modules)
    if free_vram >= target_required:
        return ResidencyPlan(
            "dual",
            f"Both models fit in VRAM — no offload expected "
            f"({free_vram / _GB:.1f} GB free, {target_required / _GB:.1f} GB needed).",
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


def _target_key_for(name: str, modules=None) -> str:
    """Best-effort cache key for a checkpoint that is not currently selected.

    Mirrors how ``main_entry.refresh_model_loading_parameters`` builds
    ``forge_loading_parameters`` -- same keys, same order, same sorted module
    list -- so the prediction matches the real key without mutating any global
    state.
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
        resolved = resolve_modules(modules)
        if resolved is None:
            resolved = current_modules()
        return str(
            dict(
                checkpoint_info=info,
                additional_modules=resolved,
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

    key = _loaded_model_key()
    if not key:
        logger.warning(
            "Model Chain: cannot identify the loaded checkpoint, so it will not be cached"
        )
        return

    if _cache.has(key):
        _cache.get(key)  # refresh recency
        return

    info = getattr(model, "sd_checkpoint_info", None)
    name = info.name_for_extra if info is not None else "(unknown)"
    size = loaded_size_bytes(model) or file_size_bytes(name)

    # What the cache may grow to right now: whatever it already holds, plus the
    # RAM actually free beyond the reserve. Expressing it this way lets admit()
    # evict older entries to make room instead of refusing outright, and keeps
    # the ceiling honest as free memory changes.
    headroom = max(free_ram_bytes() - RAM_RESERVE_BYTES, 0)
    budget = min(cache_budget_bytes(), _cache.total_bytes() + headroom)

    entry = _Entry(
        key=key,
        checkpoint_name=name,
        sd_model=model,
        size_bytes=size,
        model_flags=snapshot_model_flags(),
    )

    global _last_refusal

    if _cache.admit(entry, budget):
        _last_refusal = None
        logger.info(
            "Model Chain: holding %s in the RAM cache (%.1f GB; cache now %.1f GB of %.1f GB budget)",
            name,
            size / _GB,
            _cache.total_bytes() / _GB,
            budget / _GB,
        )
    else:
        _last_refusal = name
        logger.warning(
            "Model Chain: not enough system RAM to cache %s (%.1f GB needed, %.1f GB available to the cache) "
            "— it will reload from disk on every switch. Raise "
            '"Model Chain: max system RAM for model cache (GB)" in Settings if you have the RAM.',
            name,
            size / _GB,
            budget / _GB,
        )


def ensure_resident(name: str, modules=None) -> str:
    """Make ``name`` the loaded checkpoint, and report how it got there.

    ``modules`` is the VAE / text encoder selection for this checkpoint; None
    inherits whatever is currently selected. Because the host folds the module
    list into ``forge_loading_parameters``, a checkpoint paired with its own
    VAE and text encoder is a distinct cache entry -- which is what lets Stage 1
    and Stage 2 hold different encoders in memory at the same time.

    Returns ``"unchanged"``, ``"warm"`` (restored from the RAM cache) or
    ``"cold"`` (read from disk). This is the only entry point the orchestration
    code uses to change models; it knows nothing about slots.
    """
    from modules import processing, sd_models, shared
    from modules.sd_models import model_data
    from modules_forge import main_entry

    # A preload may still be moving Stage 1's weights. Never move models
    # underneath it -- wait, then proceed against a settled state.
    join_preload()

    info = checkpoint_info(name)
    if info is None:
        raise ModelChainError(f'Stage 2 checkpoint "{name}" was not found.')

    resolved_modules = resolve_modules(modules)

    if _is_real_model(model_data.sd_model):
        current_info = getattr(model_data.sd_model, "sd_checkpoint_info", None)
        same_checkpoint = current_info is not None and current_info.filename == info.filename
        same_modules = resolved_modules is None or resolved_modules == current_modules()
        if same_checkpoint and same_modules:
            return "unchanged"

    _stash_current()

    # Let the host recompute loading parameters, dynamic LoRA flags and the
    # checkpoint selection. save=False keeps this transient switch out of the
    # user's config.json. Both changes are made with refresh=False and followed
    # by a single refresh, the same way the host's own hires-fix pass swaps
    # checkpoint and modules together.
    main_entry.checkpoint_change(name, None, save=False, refresh=False)
    if resolved_modules is not None:
        main_entry.modules_change(resolved_modules, None, save=False, refresh=False)
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


def restore_selection(name: str, modules=None) -> None:
    """Point the UI selection and loading parameters back at Stage 1's pair.

    Both halves have to go back: restoring the checkpoint but leaving Stage 2's
    VAE and text encoder selected would silently change what Stage 1 loads next
    time, and would leave the user's VAE/TE dropdown showing Stage 2's files.

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

    main_entry.checkpoint_change(name, None, save=False, refresh=False)
    if modules is not None:
        main_entry.modules_change(modules, None, save=False, refresh=False)
    main_entry.refresh_model_loading_parameters()
    _pending_restore = name


def reinstate_pending() -> bool:
    """Warm-swap the selected checkpoint back in, if we left a different one loaded.

    Called at the top of every generation. Returns True when a swap happened.
    """
    global _pending_restore

    from modules import processing
    from modules.sd_models import model_data

    with _model_lock:
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


def last_refusal() -> str | None:
    """Name of the last model the cache refused, or None."""
    return _last_refusal


# -- Stage 1 preload ------------------------------------------------------- #
#
# Restoring Stage 1 is two separate costs. reinstate_pending() does the cheap
# half -- a pointer swap out of the RAM cache -- but the weights themselves are
# still in system RAM at that point, and the expensive half happens lazily when
# the sampler first asks for them. Deferring both to the next Generate click
# means the user waits through the whole move before a single step runs.
#
# Doing it eagerly at the end of the generation only helps if it overlaps the
# time the user spends looking at the result. Running it inline in postprocess()
# would not overlap anything: the gallery is not populated until the generation
# call returns, so an inline preload would simply move the same wait to before
# the images appear and gain nothing. Hence a background thread.
#
# The thread is joined by every entry point of this module that touches models,
# so the extension can never race itself: the worst case is that the user clicks
# Generate immediately and waits exactly as long as they would have anyway.

_preload_thread: threading.Thread | None = None
_preload_reinstated = False


def preload_async(width: int = 0, height: int = 0) -> bool:
    """Start warming Stage 1's weights back into VRAM in the background.

    ``width``/``height`` size the VRAM budget, and are the *current*
    generation's Stage 1 size because the next one's is not knowable yet. That
    only affects how much room is freed, and ``before_process`` re-checks
    against the real size before Stage 1 runs.

    Returns True if a preload was started.
    """
    global _preload_thread

    if not option(OPT_PRELOAD, True):
        return False
    if _pending_restore is None:
        return False  # nothing was swapped out, so nothing to swap back

    join_preload()
    _preload_thread = threading.Thread(
        target=_preload_worker,
        args=(width, height),
        name="model-chain-preload",
        daemon=True,
    )
    _preload_thread.start()
    return True


def join_preload(timeout: float | None = None) -> None:
    """Wait for any in-flight preload to finish."""
    global _preload_thread

    thread = _preload_thread
    if thread is None or thread is threading.current_thread():
        return

    thread.join(timeout)
    if not thread.is_alive():
        _preload_thread = None


def consume_preload() -> bool:
    """True once if a background preload already swapped Stage 1 back in.

    Lets ``before_process`` tell "nothing to do" apart from "already done", so
    the VRAM budget still gets checked against this generation's real size.
    """
    global _preload_reinstated

    was_reinstated, _preload_reinstated = _preload_reinstated, False
    return was_reinstated


def _preload_worker(width: int, height: int) -> None:
    global _preload_reinstated

    started = time.perf_counter()
    try:
        with _model_lock:
            # reinstate_pending() re-reads the live selection, so a checkpoint
            # changed in the UI since the generation ended is handled correctly
            # here: either it is cached and gets swapped in, or this returns
            # False and the next generation loads it from disk as usual.
            if not reinstate_pending():
                return

            _preload_reinstated = True

            from modules import shared

            name = shared.opts.sd_model_checkpoint
            make_vram_room(name, current_modules(), width, height, stage=STAGE_1)
            moved = _load_current_to_gpu()
    except Exception:
        logger.warning("Model Chain: Stage 1 preload failed", exc_info=True)
        return

    logger.info(
        "Model Chain: preloaded %s into VRAM in %.1fs — the next generation starts sampling immediately",
        name,
        time.perf_counter() - started,
    )
    if not moved:
        logger.debug("Model Chain: preload moved no weights; they were already resident")


def _load_current_to_gpu() -> int:
    """Move the loaded checkpoint's weights onto the GPU now.

    This is exactly the work the sampler would otherwise trigger on the next
    Generate click, done through the host's own entry point so the model ends
    up in the state normal sampling produces rather than one this extension
    invented. Returns the bytes moved, as best as free VRAM can report it.
    """
    from backend import memory_management
    from modules.sd_models import model_data

    patchers = model_patchers(model_data.sd_model)
    if not patchers:
        return 0

    before = free_vram_bytes()
    memory_management.load_models_gpu(patchers)
    return max(before - free_vram_bytes(), 0)


STAGE_1 = "Stage 1"
STAGE_2 = "Stage 2"

_pinned_patchers: list = []
"""Stage 1's text encoder and VAE, held resident across the Stage 2 switch."""


def capture_stage_1_encoders() -> int:
    """Remember Stage 1's text encoder and VAE so the Stage 2 switch can spare them.

    Called while Model A is still the loaded model, immediately before the
    switch. Only the encoders: the UNet is far too large to keep resident
    alongside Stage 2's, and it is also the one part of Model A that Stage 2
    genuinely has no use for.

    The encoders are the opposite case. They are a small fraction of a
    checkpoint but a large fraction of the *time*, because every switch pays to
    move them: a measured Krea 2 -> Flux.2 cycle spent 3.7s on Stage 1's text
    encoder against 5.6s on its UNet. Keeping them put removes that from every
    subsequent generation.

    Re-captured each generation, so pointing Stage 1 at a different checkpoint
    leaves at most one generation's worth of stale entries -- and those are
    dropped anyway by the ``current_loaded_models`` lookup in ``_pinned_keep``.

    Returns the number of patchers captured.
    """
    global _pinned_patchers

    _pinned_patchers = []
    if not option(OPT_PIN_ENCODERS, True):
        return 0

    try:
        from modules.sd_models import model_data

        _pinned_patchers = model_patchers(model_data.sd_model, attrs=("clip", "vae"))
    except Exception:
        logger.debug("Model Chain: could not capture Stage 1's encoders", exc_info=True)
        _pinned_patchers = []

    return len(_pinned_patchers)


def clear_pinned_encoders() -> None:
    """Stop sparing Stage 1's encoders."""
    global _pinned_patchers

    _pinned_patchers = []


def _loaded_entries_for(patchers: list) -> list:
    """The host's ``LoadedModel`` records for ``patchers`` that are on the GPU now.

    ``free_memory`` compares ``keep_loaded`` against its own registry entries,
    not against patchers, so the patchers have to be looked up. Matching on
    identity avoids depending on how ``LoadedModel`` defines equality.
    """
    if not patchers:
        return []

    try:
        from backend import memory_management

        registry = list(getattr(memory_management, "current_loaded_models", []))
    except Exception:
        return []

    wanted = {id(p) for p in patchers}
    return [entry for entry in registry if id(getattr(entry, "model", None)) in wanted]


def _entry_vram_bytes(entry) -> int:
    """VRAM held by one ``LoadedModel``, across host versions."""
    for attribute in ("model_loaded_memory", "model_memory"):
        method = getattr(entry, attribute, None)
        if not callable(method):
            continue
        try:
            size = int(method())
        except Exception:
            continue
        if size > 0:
            return size

    try:
        return int(entry.model.model_size())
    except Exception:
        return 0


def _loaded_target_patchers(target_name: str) -> list:
    """The target's own patchers, but only if it really is the loaded model.

    Both call sites reach ``make_vram_room`` *after* the switch, so the target
    is normally ``model_data.sd_model`` already. Verifying that rather than
    assuming it means a caller that gets the order wrong falls back to the
    conservative estimate instead of sizing the budget from the wrong model.
    """
    try:
        from modules.sd_models import model_data

        loaded = model_data.sd_model
        info = checkpoint_info(target_name)
        loaded_info = getattr(loaded, "sd_checkpoint_info", None)
        if info is None or loaded_info is None or loaded_info.filename != info.filename:
            return []
        return model_patchers(loaded)
    except Exception:
        return []


def _pass_requirement(target_name: str, modules, width: int, height: int, patchers: list) -> int:
    """VRAM the pass needs in total: the model, resident, plus its activations.

    When the target is the loaded model its patchers report their real size, so
    use that in preference to the disk estimate. The file size is a proxy that
    can be wrong in either direction -- quantised formats read larger than they
    land, mixed-precision builds land larger than they read -- and being wrong
    here either wastes an eviction or leaves the pass short.
    """
    size = 0
    for patcher in patchers:
        try:
            size += int(patcher.model_size())
        except Exception:
            size = 0
            break

    if size <= 0:
        size = int(file_size_bytes(target_name, modules) * (1.0 + VRAM_MODEL_OVERHEAD_FRACTION))

    return size + vram_headroom_bytes(width, height)


def _resident_bytes(patchers: list) -> int:
    """VRAM those patchers are already holding."""
    return sum(_entry_vram_bytes(entry) for entry in _loaded_entries_for(patchers))


def _pinned_keep(required: int, stage: str) -> tuple[list, int]:
    """Entries to spare in this eviction, and the VRAM they hold.

    Pinning is only ever applied on the way *into* Stage 2, and only when the
    incoming pass still fits with the pinned encoders in place. That condition
    is not a nicety: pinning more than the card can spare is worse than not
    pinning at all, because ``free_memory`` would fall short of its target and
    ``load_models_gpu`` would then make up the difference by partially
    unloading while it loads -- the slow path this whole function exists to
    avoid. Failing to fit therefore drops the pin rather than the target.
    """
    if stage != STAGE_2 or not _pinned_patchers:
        return [], 0

    entries = _loaded_entries_for(_pinned_patchers)
    if not entries:
        return [], 0

    pinned = sum(_entry_vram_bytes(entry) for entry in entries)
    total = total_vram_bytes()

    if total <= 0 or total - pinned < required:
        logger.info(
            "Model Chain: not pinning Stage 1's encoders — %.1f GB of VRAM less "
            "%.1f GB pinned leaves too little for the %.1f GB %s pass",
            total / _GB,
            pinned / _GB,
            required / _GB,
            stage,
        )
        return [], 0

    return entries, pinned


def make_vram_room(target_name: str, modules=None, width: int = 0, height: int = 0, stage: str = STAGE_2) -> int:
    """Evict other models from VRAM if the Stage 2 pass will not otherwise fit.

    This is the "demote only under pressure" half of the residency policy, and
    the warm-swap path is where it has to be applied explicitly. A warm swap
    deliberately skips ``unload_all_models()`` so both checkpoints can stay hot
    -- but when the incoming model plus its activations do *not* fit alongside
    the outgoing one, leaving the outgoing one resident is actively harmful:
    the host only frees what each individual ``load_models_gpu`` call asks for,
    which is far less than the whole pass needs.

    Freeing here goes through ``memory_management.free_memory``, which moves
    weights to their offload device rather than discarding them, so the evicted
    model stays in this cache and switching back to it is still warm.

    What the pass needs and what has to be *freed* are different numbers
    whenever the target is already partly on the GPU -- after a preload it may
    be entirely on the GPU. Asking to free the whole requirement in that state
    is self-defeating: the target's own weights are the biggest evictable thing
    present, so the eviction throws out exactly what the next pass is about to
    load and the load happens all over again. So the target's resident bytes are
    subtracted from the requirement, and its patchers are spared from eviction.

    Returns the number of bytes freed (0 when nothing needed doing).
    """
    own = _loaded_target_patchers(target_name)
    required = _pass_requirement(target_name, modules, width, height, own)
    resident = _resident_bytes(own)
    needed = max(required - resident, 0)
    free = free_vram_bytes()

    if free <= 0:
        return 0  # cannot query VRAM; leave the host's own management alone

    if free >= needed:
        logger.info(
            "Model Chain: %s needs %.1f GB, has %.1f GB free%s — no eviction needed",
            stage,
            required / _GB,
            free / _GB,
            f" and {resident / _GB:.1f} GB already resident" if resident else "",
        )
        return 0

    pinned_keep, pinned = _pinned_keep(required, stage)
    keep = _loaded_entries_for(own) + pinned_keep

    try:
        from backend import memory_management

        device = memory_management.get_torch_device()
        if keep:
            try:
                evicted = memory_management.free_memory(needed, device, keep_loaded=keep)
            except TypeError:
                # A host without keep_loaded is a reason to lose the exemptions,
                # never a reason to skip freeing -- falling through to no
                # eviction at all would be far worse than an unpinned encoder.
                logger.debug("Model Chain: free_memory does not take keep_loaded")
                keep, pinned = [], 0
                evicted = memory_management.free_memory(needed, device)
        else:
            evicted = memory_management.free_memory(needed, device)
    except Exception:
        logger.warning("Model Chain: failed to free VRAM for %s", stage, exc_info=True)
        return 0

    after = free_vram_bytes()
    freed = max(after - free, 0)
    names = [type(getattr(m, "model", m)).__name__ for m in (evicted or [])]

    logger.info(
        "Model Chain: freed %.1f GB VRAM for %s (%.1f GB -> %.1f GB free, %.1f GB of %.1f GB "
        "still to load)%s%s",
        freed / _GB,
        stage,
        free / _GB,
        after / _GB,
        needed / _GB,
        required / _GB,
        f"; offloaded {', '.join(names)}" if names else "",
        f"; kept {pinned / _GB:.1f} GB of Stage 1 encoders resident" if pinned_keep else "",
    )

    if after < needed:
        logger.warning(
            "Model Chain: still only %.1f GB VRAM free against %.1f GB left to load for a "
            "%dx%d pass. Expect the driver to spill into system memory, which is "
            "very slow. A smaller Stage 2 size multiplier, a lower Hires. fix "
            "upscale, or a more heavily quantised Stage 2 model would each help.",
            after / _GB,
            needed / _GB,
            width,
            height,
        )

    return freed


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
    global _pending_restore, _preload_reinstated

    join_preload()

    _pending_restore = None
    _preload_reinstated = False
    clear_pinned_encoders()
    _cache.clear()

    try:
        import gc

        from backend import memory_management

        gc.collect()
        memory_management.soft_empty_cache()
    except Exception:
        pass
