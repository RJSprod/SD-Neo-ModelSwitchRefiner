"""How much context fits, for this model, in this much VRAM.

Section 11 asks for three budgets to be kept distinct, and the distinction is
load-bearing rather than pedantic:

* **context size** is a token count asked of llama.cpp (``--ctx-size``);
* **context VRAM buffer** is the memory the residency manager sets aside for
  the key/value cache that context implies;
* **runtime/compute reserve** is neither of those -- it is the scratch space
  llama.cpp needs to evaluate a batch, and it exists whether the context is
  512 tokens or 128k.

Section 12 then asks that the first be predicted from the second per model,
"not a generic tokens per GB constant". The arithmetic below is therefore
built from the file's own header (see ``mc_gguf``):

    kv bytes per token = blocks x kv heads x (key width + value width) x element size

which is why a grouped-query model with 8 KV heads costs a quarter of what a
32-head model of the same width costs, and why a constant would be wrong for
both. Both halves of the cache are sized separately because llama.cpp lets
them be quantised separately (``--cache-type-k`` / ``--cache-type-v``).

What is estimated and what is measured
--------------------------------------
Weights and KV cache are arithmetic and are close to exact. The compute buffer
is not: it depends on the batch size, the backend, and the graph llama.cpp
plans, and any formula for it here would be a guess with a decimal point.

So it starts as a coarse allowance and is then *replaced by measurement*. The
first real load of a given model/placement records what the card actually lost,
and every estimate afterwards for that same configuration uses the observed
overhead instead of the guess -- which is what section 12's "whether the
estimate is theoretical or calibrated from observed runtime behaviour" is
asking the UI to be able to say. :attr:`Estimate.calibrated` is that flag.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import mc_gguf

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3
_MB = 1024**2


# --------------------------------------------------------------------------- #
# KV cache element sizes
# --------------------------------------------------------------------------- #
#
# Bytes per element for the cache types llama.cpp accepts on --cache-type-k /
# --cache-type-v. The quantised ones are block formats: q8_0 packs 32 values
# into 34 bytes (one fp16 scale plus 32 bytes), q4_0 into 18, and so on, which
# is where the fractions come from.

KV_TYPES: tuple[tuple[str, float, str], ...] = (
    ("f32", 4.0, "f32 — full precision, twice the cost of f16"),
    ("f16", 2.0, "f16 — llama.cpp's default"),
    ("bf16", 2.0, "bf16 — same size as f16"),
    ("q8_0", 34 / 32, "q8_0 — about half of f16, negligible quality cost"),
    ("q5_1", 24 / 32, "q5_1"),
    ("q5_0", 22 / 32, "q5_0"),
    ("q4_1", 20 / 32, "q4_1"),
    ("q4_0", 18 / 32, "q4_0 — about a quarter of f16, measurable quality cost"),
)

KV_TYPE_BYTES = {name: size for name, size, _ in KV_TYPES}
KV_TYPE_LABELS = [(name, label) for name, _, label in KV_TYPES]

DEFAULT_KV_TYPE = "f16"

CONTEXT_GRANULARITY = 256
"""Recommended figures are rounded down to a multiple of this.

A recommendation of 31,847 tokens invites the reader to believe the estimate is
accurate to the token. It is not, and rounding says so.
"""

SAFE_FRACTION = 0.85
"""Share of a theoretical capacity offered as the recommended one.

The gap absorbs what the arithmetic cannot see: fragmentation, the difference
between a planned graph and an executed one, and the fact that a card with
exactly enough memory has none left for the driver to breathe. Section 12 asks
for both numbers to be shown rather than for one to be quietly reduced.
"""

COMPUTE_BASE_BYTES = 192 * _MB
COMPUTE_PER_EMBD_BYTES = 48 * 1024
"""The a-priori compute allowance: a fixed floor plus a term in model width.

Both are round numbers and neither is defended as precise -- a 4096-wide model
lands near 380 MB, which is the right order for llama.cpp's CUDA compute buffer
at its default batch size. Calibration replaces this the first time the model
is actually loaded; until then it errs high, because an overestimate costs one
avoidable eviction and an underestimate costs a spill into system memory.
"""


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #


ALL_LAYERS = -1
"""``gpu_layers`` meaning "offload everything", llama.cpp's own default here."""

NO_LAYERS = 0
"""Weights stay in system RAM: mixed mode, and CPU mode with it."""

ALL_EXPERTS = -1
"""``cpu_expert_layers`` meaning "every expert tensor in system RAM".

llama.cpp's ``--cpu-moe``, and the last rung of the expert ladder rather than
its only one -- see :attr:`Placement.cpu_expert_layers`. Spelled as a negative
sentinel for the same reason :data:`ALL_LAYERS` is: "all" is not a count, and a
count large enough to mean all is a count that goes wrong the first time a
model with more blocks than the number arrives.
"""

NO_EXPERTS = 0
"""Every expert on the card, which is where they belong when there is room."""


@dataclass(frozen=True)
class Placement:
    """Where an LLM is to be put, in the terms the estimate depends on."""

    gpu_layers: int = ALL_LAYERS
    context: int = 8192
    kv_type_k: str = DEFAULT_KV_TYPE
    kv_type_v: str = DEFAULT_KV_TYPE
    on_gpu: bool = True
    """False for CPU mode -- no VRAM is involved at all, and every figure
    below is then zero rather than small."""
    cpu_expert_layers: int = NO_EXPERTS
    """How many blocks keep their expert tensors in system RAM.

    Moving experts is the right way to make a mixture-of-experts model smaller
    on the card, and it is not "fewer layers". Experts are the great majority of
    the weights and are consulted a couple at a time; attention is small and is
    touched by every token. Dropping whole blocks moves both, so it gives up
    attention -- the part that most wants to be resident -- to save weights that
    are mostly idle. Moving only the experts saves nearly the same VRAM and
    keeps every block's attention on the card.

    This used to be a boolean, and a boolean made it an all-or-nothing choice: a
    model two gigabytes too large for a card moved *every* expert into system
    RAM, when the first six blocks' worth would have been enough. So it is a
    count now, and the three states it can be in are worth naming:

    ``NO_EXPERTS``
        Nothing moved. No expert flag reaches the command line.
    ``N > 0``
        ``--n-cpu-moe N``: the first ``N`` blocks keep their experts in system
        RAM and every other tensor, in every block, stays on the card.
    ``ALL_EXPERTS``
        ``--cpu-moe``: every expert in system RAM, which is where the ladder
        ends before whole blocks start to leave.

    Only meaningful for a model with experts, and only when the runtime
    understands the corresponding flag; both are checked before this is set.
    """

    def with_context(self, context: int) -> "Placement":
        from dataclasses import replace

        return replace(self, context=int(context))

    def with_layers(self, layers: int) -> "Placement":
        from dataclasses import replace

        return replace(self, gpu_layers=int(layers))

    def with_cpu_expert_layers(self, layers: int) -> "Placement":
        """The same placement with ``layers`` blocks' experts in system RAM."""
        from dataclasses import replace

        layers = int(layers)
        return replace(self, cpu_expert_layers=layers if layers >= 0 else ALL_EXPERTS)

    def with_cpu_experts(self, cpu_experts: bool = True) -> "Placement":
        """All experts in system RAM, or none. The boolean this field used to be.

        Kept because "move every expert" is still a real rung of the ladder --
        it is the one taken when the build has ``--cpu-moe`` and not
        ``--n-cpu-moe``, and the one the progressive search lands on when every
        block has to give its experts up anyway.
        """
        return self.with_cpu_expert_layers(ALL_EXPERTS if cpu_experts else NO_EXPERTS)

    @property
    def cpu_experts(self) -> bool:
        """Whether any expert tensor is in system RAM at all.

        The compatibility shim for the boolean this replaced, and the only
        question most callers are actually asking.
        """
        return self.cpu_expert_layers != NO_EXPERTS

    @property
    def all_cpu_experts(self) -> bool:
        """Whether *every* expert is in system RAM -- llama.cpp's ``--cpu-moe``."""
        return self.cpu_expert_layers == ALL_EXPERTS

    @property
    def experts_key(self) -> str:
        """The expert split as one token, in llama.cpp's own vocabulary."""
        if self.all_cpu_experts:
            return "cpu-moe"
        return f"ncmoe-{max(int(self.cpu_expert_layers), 0)}"

    @property
    def key(self) -> str:
        """Identity for calibration: everything that changes the footprint
        except the context, which the calibration arithmetic divides out.

        The expert split is part of it because it is part of the footprint: two
        placements that differ only in how many blocks left their experts behind
        are two different amounts of VRAM, and a measurement of one filed under
        the other is worse than no measurement at all.
        """
        return (f"{self.gpu_layers}/{self.kv_type_k}/{self.kv_type_v}/"
                f"{int(self.on_gpu)}/{self.experts_key}")

    @property
    def speed_token(self) -> str:
        """This placement as one key-safe word, for keying measured speed by.

        Tokens a second is a property of *where the model ran*, not of which
        model ran: the same backbone writes at forty tokens a second resident
        and at five from system RAM, and a store that averages the two answers
        neither question. So the measurement is keyed by this.

        ``gpu``, ``ncmoe-8``, ``cpu-moe`` and ``cpu`` are the four the design
        intent names. The layer suffix is the fifth case it does not, because
        the ladder can reach it: once every expert has moved and blocks start
        leaving too, "on the GPU" covers placements an order of magnitude apart.
        """
        if not self.on_gpu or self.gpu_layers == NO_LAYERS:
            return "cpu"
        token = "gpu" if self.cpu_expert_layers == NO_EXPERTS else self.experts_key
        if self.gpu_layers != ALL_LAYERS and self.gpu_layers > 0:
            token = f"{token}-l{self.gpu_layers}"
        return token

    def describe(self, total_layers: int = 0) -> str:
        experts = self._describe_experts(total_layers)
        if not self.on_gpu:
            return "system RAM (no GPU offload)"
        if self.gpu_layers == ALL_LAYERS:
            return f"all layers on the GPU{experts}"
        if self.gpu_layers <= 0:
            return "no layers on the GPU (weights in system RAM)"
        if total_layers:
            return f"{self.gpu_layers} of {total_layers} layers on the GPU{experts}"
        return f"{self.gpu_layers} layers on the GPU{experts}"

    def _describe_experts(self, total_layers: int = 0) -> str:
        if self.cpu_expert_layers == NO_EXPERTS:
            return ""
        if self.all_cpu_experts:
            return ", experts in system RAM"
        of_total = f" of {total_layers}" if total_layers else ""
        return (f", the experts of {self.cpu_expert_layers}{of_total} "
                f"layers in system RAM")


# --------------------------------------------------------------------------- #
# The estimate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Estimate:
    """What one placement of one model is expected to cost, and what fits."""

    model: Path
    context: int
    ceiling: int
    """The model's own declared maximum context. 0 when the header did not say."""
    weights_bytes: int
    kv_bytes: int
    compute_bytes: int
    kv_bytes_per_token: float
    calibrated: bool
    placement: Placement
    detail: str = ""
    """Why the numbers are what they are, or why they are coarse."""

    @property
    def total_bytes(self) -> int:
        return int(self.weights_bytes + self.kv_bytes + self.compute_bytes)

    @property
    def resident_bytes(self) -> int:
        """What stays on the card between requests: weights and cache.

        The compute buffer is allocated too, so this is not the whole footprint
        -- it is the part that a fit decision must assume persists.
        """
        return int(self.weights_bytes + self.kv_bytes)

    @property
    def capped(self) -> bool:
        """Whether the model's own ceiling, not VRAM, is what limits context."""
        return bool(self.ceiling) and self.context >= self.ceiling


@dataclass(frozen=True)
class Capacity:
    """How much context a given VRAM budget buys (section 12's table row)."""

    budget_bytes: int
    theoretical: int
    recommended: int
    ceiling: int
    limited_by_model: bool
    calibrated: bool

    @property
    def usable(self) -> int:
        """What the UI should actually offer: never past the model's ceiling."""
        if self.ceiling:
            return min(self.recommended, self.ceiling)
        return self.recommended


def kv_bytes_per_token(gguf: mc_gguf.Gguf, placement: Placement) -> float:
    """Key/value cache cost of one token, in VRAM, for this placement.

    Summed per block rather than multiplied by a block count, because the
    blocks are not all the same shape: a hybrid model interleaves blocks that
    keep no cache at all with blocks that do, and an interleaved local/global
    design gives them different widths. Multiplying the widest block by the
    block count overstates such a model's cache by a factor of two or more,
    which shows up as a context ceiling far lower than the card can actually
    hold. See ``mc_gguf.Gguf.head_counts_kv``.

    Only offloaded blocks keep their cache on the card, and llama.cpp offloads
    the *last* ``--n-gpu-layers`` of them, so a partial offload is costed from
    the end of the model. That is the mechanism behind section 13's graceful
    degradation: reducing layers reduces both halves of the footprint at once.
    """
    if not placement.on_gpu or not gguf.usable:
        return 0.0

    k_size = KV_TYPE_BYTES.get(placement.kv_type_k, 2.0)
    v_size = KV_TYPE_BYTES.get(placement.kv_type_v, 2.0)
    per_block = [heads * (key * k_size + value * v_size)
                 for heads, key, value
                 in zip(gguf.head_counts_kv, gguf.key_lengths, gguf.value_lengths)]

    if placement.gpu_layers != ALL_LAYERS:
        offloaded = max(min(int(placement.gpu_layers), len(per_block)), 0)
        per_block = per_block[len(per_block) - offloaded:] if offloaded else []
    return float(sum(per_block))


def expert_fraction_moved(gguf: mc_gguf.Gguf, placement: Placement) -> float:
    """What share of the whole file's bytes this expert split leaves off the card.

    The arithmetic the design intent asks for, and it is deliberately the
    coarsest thing that is still right in the two cases that matter::

        expert_fraction_moved = expert_share * clamp(N / block_count, 0, 1)

    Experts are assumed to be spread evenly across the blocks, which is true of
    every published mixture-of-experts this catalogue names and is why ``N``
    blocks' experts are ``N / block_count`` of them. ``ALL_EXPERTS`` skips the
    proration entirely rather than relying on ``N == block_count``, because
    ``--cpu-moe`` moves every expert whatever this file thinks the block count
    is.

    Pre-load planning only. The observed footprint recorded after a real start
    remains the authoritative number for any configuration actually run.
    """
    share = float(getattr(gguf, "expert_share", 0.0) or 0.0)
    layers = int(getattr(placement, "cpu_expert_layers", NO_EXPERTS))
    if share <= 0 or layers == NO_EXPERTS:
        return 0.0
    if layers == ALL_EXPERTS:
        return share

    blocks = int(getattr(gguf, "block_count", 0) or 0)
    if blocks <= 0:
        # No block count to prorate against. Charging for the whole expert share
        # would under-state the footprint of a partial split, and under-stating
        # it is the direction that ends in an out-of-memory error rather than in
        # an avoidable eviction -- so nothing is discounted at all.
        return 0.0
    return share * min(max(layers / blocks, 0.0), 1.0)


def weights_bytes(gguf: mc_gguf.Gguf, placement: Placement) -> int:
    """VRAM the weights themselves take under this placement.

    The file size is the whole model, and a full offload puts effectively all
    of it on the card. A partial offload is prorated by block, which is an
    approximation and is flagged as one: the token embedding and output head
    are not repeating blocks and do not move at the same threshold llama.cpp
    moves the blocks at. It is close enough to plan an eviction against and is
    superseded by calibration for any configuration actually run.
    """
    if not placement.on_gpu:
        return 0
    # Whatever share of the experts stayed behind is VRAM the card does not
    # spend. Estimated from the header rather than measured -- see
    # ``mc_gguf.Gguf.expert_share`` and :func:`expert_fraction_moved` -- and
    # superseded by the load report llama.cpp writes once the server is up.
    resident = max(1.0 - expert_fraction_moved(gguf, placement), 0.0)
    if placement.gpu_layers == ALL_LAYERS:
        return int(gguf.file_bytes * resident)
    blocks = gguf.block_count
    if blocks <= 0 or placement.gpu_layers <= 0:
        return 0
    share = min(max(int(placement.gpu_layers), 0), blocks) / blocks
    return int(gguf.file_bytes * share * resident)


def compute_bytes(gguf: mc_gguf.Gguf, placement: Placement) -> int:
    """Runtime scratch, separate from weights and from the cache (section 11)."""
    if not placement.on_gpu or placement.gpu_layers == NO_LAYERS:
        return 0
    return int(COMPUTE_BASE_BYTES + COMPUTE_PER_EMBD_BYTES * max(gguf.embedding_length, 0))


def estimate(model: str | Path, placement: Placement,
             gguf: mc_gguf.Gguf | None = None) -> Estimate:
    """Everything section 12 asks the UI to be able to show, for one placement."""
    path = Path(model).expanduser()
    described = gguf if gguf is not None else mc_gguf.describe(path)

    if described is None or not described.usable:
        # No header, or one without the attention keys. The file size is still
        # a real number and the weights are still most of the footprint, so the
        # estimate degrades to that rather than to nothing -- and says so.
        size = 0
        try:
            size = path.stat().st_size
        except OSError:
            pass
        return Estimate(
            model=path, context=int(placement.context), ceiling=0,
            weights_bytes=int(size if placement.on_gpu else 0), kv_bytes=0,
            compute_bytes=0, kv_bytes_per_token=0.0, calibrated=False,
            placement=placement,
            detail=("This file's GGUF header does not describe its attention shape, so only "
                    "its weights can be sized. Context cost is unknown."),
        )

    per_token = kv_bytes_per_token(described, placement)
    weights = weights_bytes(described, placement)
    overhead, calibrated = _overhead(described, placement)
    kv = int(per_token * max(int(placement.context), 0))

    detail = ""
    if placement.gpu_layers not in (ALL_LAYERS,) and placement.on_gpu:
        detail = ("Partial offload: the weight figure is prorated per block and does not "
                  "model the embedding and output tensors exactly.")
    return Estimate(
        model=path, context=int(placement.context), ceiling=described.context_length,
        weights_bytes=weights, kv_bytes=kv, compute_bytes=int(overhead),
        kv_bytes_per_token=per_token, calibrated=calibrated, placement=placement,
        detail=detail,
    )


def capacity(model: str | Path, placement: Placement, budget_bytes: int,
             gguf: mc_gguf.Gguf | None = None) -> Capacity:
    """How much context ``budget_bytes`` of *context buffer* buys.

    The budget is the context/KV buffer alone -- not the weights, which are
    sized separately, and not the compute reserve, which is subtracted here
    before any tokens are counted. Keeping them apart is section 11's whole
    point: a user who raises the context buffer expects to buy context with it,
    not to discover the increase went on scratch space.
    """
    described = gguf if gguf is not None else mc_gguf.describe(Path(model).expanduser())
    per_token = kv_bytes_per_token(described, placement) if described is not None else 0.0
    ceiling = described.context_length if described is not None else 0

    if per_token <= 0:
        return Capacity(int(budget_bytes), 0, 0, ceiling, False, False)

    _, calibrated = _overhead(described, placement)
    theoretical = int(max(int(budget_bytes), 0) / per_token)
    recommended = int(theoretical * SAFE_FRACTION) // CONTEXT_GRANULARITY * CONTEXT_GRANULARITY
    limited = bool(ceiling) and recommended > ceiling
    return Capacity(int(budget_bytes), theoretical, max(recommended, 0), ceiling, limited, calibrated)


def table(model: str | Path, placement: Placement,
          buffers_gb: tuple[float, ...] = (1, 2, 4, 6, 8, 12, 16)) -> list[Capacity]:
    """Section 12's "buffer -> estimated context -> recommended context" table."""
    described = mc_gguf.describe(Path(model).expanduser())
    return [capacity(model, placement, int(gb * _GB), gguf=described) for gb in buffers_gb]


def context_for_budget(model: str | Path, placement: Placement, budget_bytes: int) -> int:
    """The context to actually ask llama.cpp for, given a buffer. Never past the ceiling."""
    found = capacity(model, placement, budget_bytes)
    return max(found.usable, 0)


def automatic_buffer_bytes(free_vram: int, weights: int, reserve: int) -> int:
    """The context buffer to use when the user chose automatic sizing.

    Everything left on the card once the weights, the runtime reserve and the
    global safety margin are accounted for -- and never negative, because a
    negative buffer is a way of saying the model does not fit, which is a
    decision for the broker rather than a number for the estimator.
    """
    return max(int(free_vram) - int(weights) - int(reserve), 0)


# --------------------------------------------------------------------------- #
# Calibration (section 12, last paragraph)
# --------------------------------------------------------------------------- #
#
# The estimate above is arithmetic plus one guess. This replaces the guess with
# a measurement, per model and per placement, as soon as there has been one.
#
# What is recorded is the *overhead*: observed total, minus the weights and
# cache the arithmetic already accounts for exactly. Storing the total instead
# would be useless the moment the context changed, because the cache term
# scales with context and the overhead does not.

FILENAME = "model_chain_llm_calibration.json"

_calibration: dict | None = None
_calibration_lock = threading.RLock()


def _store_path() -> Path:
    try:
        import mc_llm_paths

        return mc_llm_paths.data_root() / "data" / FILENAME
    except Exception:
        return Path(tempfile.gettempdir()) / FILENAME


def _signature(gguf: mc_gguf.Gguf | None, placement: Placement) -> str:
    if gguf is None:
        return ""
    return f"{gguf.path.name}:{gguf.file_bytes}:{placement.key}"


def _load() -> dict:
    global _calibration

    with _calibration_lock:
        if _calibration is not None:
            return _calibration
        try:
            _calibration = json.loads(_store_path().read_text(encoding="utf-8"))
            if not isinstance(_calibration, dict):
                _calibration = {}
        except (OSError, ValueError):
            _calibration = {}
        return _calibration


def _overhead(gguf: mc_gguf.Gguf | None, placement: Placement) -> tuple[int, bool]:
    """``(bytes, calibrated)`` for the runtime reserve of this configuration."""
    if gguf is None:
        return 0, False
    recorded = _load().get(_signature(gguf, placement))
    if isinstance(recorded, (int, float)) and recorded > 0:
        return int(recorded), True
    return compute_bytes(gguf, placement), False


def record_observation(model: str | Path, placement: Placement, observed_bytes: int) -> bool:
    """Fold a real measurement into the estimate for this configuration.

    ``observed_bytes`` is the VRAM the card actually lost across the load. The
    weights and cache the arithmetic already predicts are subtracted, and what
    is left is the overhead worth remembering.

    Returns whether anything was stored. A measurement smaller than the
    predicted weights alone is discarded rather than clamped: it means
    something else freed memory during the load, not that the runtime is free.
    """
    described = mc_gguf.describe(Path(model).expanduser())
    if described is None or not described.usable:
        return False

    predicted = weights_bytes(described, placement) + int(
        kv_bytes_per_token(described, placement) * max(placement.context, 0))
    overhead = int(observed_bytes) - predicted
    if overhead <= 0 or overhead > 8 * _GB:
        logger.debug("Model Chain: discarding implausible LLM calibration (%s bytes overhead)",
                     overhead)
        return False

    with _calibration_lock:
        store = dict(_load())
        store[_signature(described, placement)] = overhead
        _write(store)
        globals()["_calibration"] = store
    logger.info("Model Chain: calibrated LLM runtime reserve for %s at %.0f MB",
                described.path.name, overhead / _MB)
    return True


def forget() -> None:
    """Drop every recorded measurement. For tests, and for a settings reset."""
    with _calibration_lock:
        globals()["_calibration"] = {}
        _write({})


def _write(store: dict) -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(store, stream, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except OSError:
        logger.debug("Model Chain: could not write LLM calibration", exc_info=True)
