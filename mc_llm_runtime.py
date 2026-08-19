"""The llama.cpp runtime, placed by the residency broker.

Section 6 says keep llama.cpp in its own process, and that is what the vendored
``prompt_master.inference`` package already does. This module is the layer
between that process and ModelSwitch's memory policy. Its whole job is to
answer one question before every load:

    given what is on the card right now, and what the user asked for, where
    does this model actually go?

and then to say out loud what it decided, because section 13 forbids quietly
reducing context or quality-critical settings without reporting the change.

Why a separate process is the right shape here
----------------------------------------------
It is an ownership boundary, not an implementation detail. llama.cpp allocates
CUDA memory that PyTorch does not know about and cannot move, so the only
reliable way to hand VRAM back to Forge is to end the process that holds it.
That is a crude mechanism, and it is also an *honest* one: when this module
tells the broker it freed 14 GB, the 14 GB is genuinely gone rather than
cached somewhere by an allocator.

The warm layer, for a model in another process
----------------------------------------------
Section 7.2 asks that the intent of a warm tier survive even where the
mechanism differs, and for llama.cpp it differs completely: there is no
"move these weights to system RAM" call to make. What there is instead is the
operating system's page cache. llama.cpp reads a GGUF through ``mmap``, so
after one load the file's pages are resident in system RAM, and stopping the
server does not evict them. A restart then reads from RAM at memory bandwidth
rather than from disk -- which is the warm tier, arrived at by a different
road. Nothing here has to do anything to get it except *not* do the one thing
that would lose it, which is to thrash the machine's RAM so hard the kernel
drops the pages. That is why :func:`release` prefers stopping the server to
reloading it somewhere else.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mc_broker
import mc_gguf
import mc_llm_context
import mc_llm_paths
import mc_llm_state

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3
_MB = 1024**2

OPT_RELEASE = "model_chain_llm_release"

RELEASE_STOP = "stop"
RELEASE_SYSTEM_RAM = "system_ram"

RELEASE_MODES = (
    (RELEASE_STOP, "Stop llama-server — releases all VRAM; weights stay warm in the page cache"),
    (RELEASE_SYSTEM_RAM, "Keep it running in system RAM — slower inference, no reload"),
)

RELEASE_LOCK_TIMEOUT = 30.0
"""Seconds :meth:`Runtime.release` waits for the runtime lock before giving up.

Long enough to outlast anything but a model load, short enough that a wedged
one delays a generation rather than joining it. See the note in ``release``.
"""

MINIMUM_CONTEXT = 2048
"""The floor placement negotiation will shrink context to.

Below this the LLM is not usefully a chat model any more -- a system prompt and
one image already spend a good part of it -- so a placement that would need to
go lower is reported as not fitting rather than silently made useless.
"""


class NotConfigured(RuntimeError):
    """No model has been chosen yet. The panel offers setup rather than an error."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """What the installation is set up to run, from the state file and prefs."""

    runtime: Path | None
    model: Path | None
    mmproj: Path | None
    gpu_index: int
    device: str
    gpu_layers: str
    context_size: int
    context_mode: str
    context_buffer_gb: float
    kv_type_k: str
    kv_type_v: str
    quantization: str = ""
    device_name: str = ""
    mode: str = "gpu"

    @property
    def on_gpu(self) -> bool:
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD

        return self.device.casefold() != CPU_DEVICE and str(self.gpu_layers) != NO_OFFLOAD

    @property
    def configured(self) -> bool:
        return self.model is not None and self.runtime is not None

    @property
    def sees(self) -> bool:
        return self.mmproj is not None


def config() -> Config:
    """Current runtime configuration. Never raises -- an unconfigured install
    is a state the panel renders, not an exception it handles."""
    from prompt_master.core.config import read_json

    paths = mc_llm_paths.app_paths()
    try:
        state = read_json(paths.state_file)
    except (OSError, ValueError):
        state = {}
    prefs = mc_llm_state.preferences()

    def located(key):
        recorded = str(state.get(key) or "")
        if not recorded:
            return None
        try:
            return paths.locate(recorded)
        except ValueError:
            return None

    runtime = None
    recorded_runtime = str(state.get("runtime") or "")
    if recorded_runtime:
        try:
            runtime = paths.contained(recorded_runtime)
        except ValueError:
            runtime = None

    return Config(
        runtime=runtime,
        model=located("model"),
        mmproj=located("mmproj"),
        gpu_index=int(state.get("gpu_index", 0) or 0),
        device=str(state.get("gpu_device", "CUDA0")),
        gpu_layers=str(state.get("gpu_layers", "all")),
        context_size=int(prefs.get("context_size") or state.get("context_size", 8192) or 8192),
        context_mode=str(prefs.get("context_mode", "auto")),
        context_buffer_gb=float(prefs.get("context_buffer_gb", 4.0) or 0.0),
        kv_type_k=str(prefs.get("kv_type_k", "f16")),
        kv_type_v=str(prefs.get("kv_type_v", "f16")),
        quantization=str(state.get("quantization", "")),
        device_name=str(state.get("gpu_device_name", state.get("gpu_name", ""))),
        mode=str(state.get("mode", "gpu")),
    )


def _release_mode() -> str:
    return mc_broker.resolve(mc_broker.option(OPT_RELEASE, RELEASE_STOP),
                             RELEASE_MODES, RELEASE_STOP)


# --------------------------------------------------------------------------- #
# Placement negotiation (sections 11 and 13)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Negotiation:
    """A placement, and every change made to reach it.

    ``notes`` is not decoration. Section 13: "The app must not quietly reduce
    context or quality-critical settings without reporting what it changed."
    Every branch below that lowers something appends a sentence here, and the
    UI prints them.
    """

    placement: mc_llm_context.Placement
    estimate: mc_llm_context.Estimate
    notes: tuple[str, ...] = ()
    fits: bool = True

    @property
    def degraded(self) -> bool:
        return bool(self.notes)


def negotiate(configuration: Config | None = None,
              gguf: mc_gguf.Gguf | None = None, *, reclaim: bool = True) -> Negotiation:
    """Decide where the LLM goes, given what is on the card right now.

    ``reclaim=False`` answers the same question without moving anything, which
    is what the estimator panel asks. That distinction is not a nicety: the
    panel is rendered when the tab is built and every time somebody opens the
    accordion, and a preview that evicted a checkpoint to show a table would be
    a far worse bug than any it was drawing attention to.

    The ladder, in order, and it is the order that encodes the policy:

    1. Ask for what the user configured. If it fits in what is already free,
       stop -- nothing moves, which is section 8's rule and the common case in
       Hybrid mode once both models are small enough.
    2. Under **Preserve image**, never ask the image side for anything: shrink
       the context, then the offload, until it fits or it cannot.
    3. Under **Adaptive**, shrink the context to its floor first and only then
       ask the broker for room. Reducing a context buffer nobody is using is
       less disruptive than evicting a checkpoint somebody is about to use,
       which is what "least disruption" has to mean if it means anything.
    4. Under **LLM priority**, ask the broker first and shrink only if the
       broker could not find enough.
    5. In **Exclusive** mode the broker sweeps the image family out entirely
       before any of this, so step 1 usually succeeds on its own.
    """
    configuration = configuration or config()
    if not configuration.configured:
        raise NotConfigured(
            "No local model is configured yet. Choose a GGUF in LLM Studio’s Setup mode."
        )

    described = gguf if gguf is not None else mc_gguf.describe(configuration.model)
    wanted = _requested_placement(configuration, described)
    notes: list[str] = []

    if not wanted.on_gpu:
        # System RAM or CPU execution: there is no VRAM decision to make, and
        # pretending to make one would report movements that never happened.
        return Negotiation(wanted, mc_llm_context.estimate(configuration.model, wanted, described),
                           (), True)

    reserve = mc_broker.safety_margin_bytes()
    estimate = mc_llm_context.estimate(configuration.model, wanted, described)

    if reclaim and mc_broker.mode() == mc_broker.MODE_EXCLUSIVE:
        mc_broker.request_vram(mc_broker.FAMILY_LLM, estimate.total_bytes,
                               reason="LLM took VRAM ownership (Exclusive mode)", margin=reserve)
        estimate = mc_llm_context.estimate(configuration.model, wanted, described)

    if _fits(estimate, reserve):
        return Negotiation(wanted, estimate, (), True)

    chosen = mc_broker.policy()
    placement = wanted

    if chosen == mc_broker.POLICY_LLM_PRIORITY:
        placement, estimate, freed = _ask_broker(configuration, placement, described, reserve,
                                                 reclaim)
        if freed:
            notes.append(freed)
        if _fits(estimate, reserve):
            return Negotiation(placement, estimate, tuple(notes), True)

    placement, estimate, shrunk = _shrink_context(configuration, placement, described, reserve)
    if shrunk:
        notes.append(shrunk)
    if _fits(estimate, reserve):
        return Negotiation(placement, estimate, tuple(notes), True)

    if chosen == mc_broker.POLICY_ADAPTIVE:
        placement, estimate, freed = _ask_broker(configuration, placement, described, reserve,
                                                 reclaim)
        if freed:
            notes.append(freed)
        if _fits(estimate, reserve):
            return Negotiation(placement, estimate, tuple(notes), True)

    placement, estimate, offloaded = _shrink_offload(configuration, placement, described, reserve)
    if offloaded:
        notes.append(offloaded)

    return Negotiation(placement, estimate, tuple(notes), _fits(estimate, reserve))


def _requested_placement(configuration: Config,
                         gguf: mc_gguf.Gguf | None) -> mc_llm_context.Placement:
    """The placement the user's settings ask for, before any negotiation."""
    from prompt_master.inference.device_detection import NO_OFFLOAD

    layers = mc_llm_context.ALL_LAYERS
    recorded = str(configuration.gpu_layers).strip().casefold()
    if recorded == NO_OFFLOAD:
        layers = mc_llm_context.NO_LAYERS
    elif recorded not in ("all", "-1", ""):
        try:
            layers = int(recorded)
        except ValueError:
            layers = mc_llm_context.ALL_LAYERS

    placement = mc_llm_context.Placement(
        gpu_layers=layers,
        context=max(int(configuration.context_size), MINIMUM_CONTEXT),
        kv_type_k=configuration.kv_type_k,
        kv_type_v=configuration.kv_type_v,
        on_gpu=configuration.on_gpu,
    )

    if configuration.context_mode == "auto" and placement.on_gpu:
        # Automatic sizing: spend what is free after weights and reserves on
        # context, rather than whatever number the state file happens to hold.
        weights = mc_llm_context.weights_bytes(gguf, placement) if gguf is not None else 0
        budget = mc_llm_context.automatic_buffer_bytes(
            mc_broker.free_vram_bytes(), weights, mc_broker.safety_margin_bytes())
        sized = mc_llm_context.context_for_budget(configuration.model, placement, budget)
        if sized >= MINIMUM_CONTEXT:
            placement = placement.with_context(sized)

    return _capped(placement, gguf)


def _capped(placement: mc_llm_context.Placement,
            gguf: mc_gguf.Gguf | None) -> mc_llm_context.Placement:
    """Never ask for more context than the model itself declares (section 12)."""
    if gguf is None or not gguf.context_length:
        return placement
    if placement.context <= gguf.context_length:
        return placement
    return placement.with_context(gguf.context_length)


def _fits(estimate: mc_llm_context.Estimate, reserve: int) -> bool:
    return mc_broker.free_vram_bytes() >= estimate.total_bytes + reserve


def _ask_broker(configuration: Config, placement, gguf, reserve: int, reclaim: bool = True):
    """Ask the image side for room, or -- when previewing -- do not ask at all.

    A preview that declined to demote reports a placement that fits less well
    than the real one would, which is the right way round: the panel may
    understate what is possible, and must never take VRAM from a workload to
    populate a table.
    """
    estimate = mc_llm_context.estimate(configuration.model, placement, gguf)
    if not reclaim:
        return placement, estimate, ""
    released = mc_broker.request_vram(mc_broker.FAMILY_LLM, estimate.total_bytes,
                                      reason="an LLM request", margin=reserve,
                                      exclusive_sweep=False)
    estimate = mc_llm_context.estimate(configuration.model, placement, gguf)
    return placement, estimate, (released.describe() if released.moved_anything else "")


def _shrink_context(configuration: Config, placement, gguf, reserve: int):
    """Lower the context until the cache fits what is free, or hit the floor."""
    free = mc_broker.free_vram_bytes()
    estimate = mc_llm_context.estimate(configuration.model, placement, gguf)
    per_token = estimate.kv_bytes_per_token
    if per_token <= 0:
        return placement, estimate, ""

    budget = free - reserve - estimate.weights_bytes - estimate.compute_bytes
    affordable = int(max(budget, 0) / per_token)
    affordable = affordable // mc_llm_context.CONTEXT_GRANULARITY * mc_llm_context.CONTEXT_GRANULARITY

    if affordable >= placement.context:
        return placement, estimate, ""
    if affordable < MINIMUM_CONTEXT:
        return placement, estimate, ""

    was = placement.context
    placement = placement.with_context(affordable)
    return (placement, mc_llm_context.estimate(configuration.model, placement, gguf),
            f"context reduced from {was:,} to {affordable:,} tokens to fit the free VRAM")


def _shrink_offload(configuration: Config, placement, gguf, reserve: int):
    """Move blocks off the card, which is section 13's graceful degradation.

    Reached only when a full-precision placement will not fit even at the
    minimum context. Every block left in system RAM costs speed and gives back
    both its weights and its share of the cache, so this converges quickly --
    and if it converges all the way to zero the model runs from system RAM,
    which is slow but is still an answer rather than a failure.
    """
    if gguf is None or not gguf.usable or gguf.block_count <= 0:
        return placement, mc_llm_context.estimate(configuration.model, placement, gguf), ""

    total = gguf.block_count
    free = mc_broker.free_vram_bytes()
    for layers in range(total - 4, -1, -4):
        candidate = placement.with_layers(max(layers, 0))
        estimate = mc_llm_context.estimate(configuration.model, candidate, gguf)
        if free >= estimate.total_bytes + reserve:
            note = (f"offload reduced to {max(layers, 0)} of {total} layers on the GPU; "
                    "the rest run from system RAM and will be slower")
            if layers <= 0:
                note = ("the whole model was placed in system RAM -- there is not enough free "
                        "VRAM for any of it alongside what is resident; generation will be slow")
            return candidate, estimate, note

    candidate = placement.with_layers(mc_llm_context.NO_LAYERS)
    return (candidate, mc_llm_context.estimate(configuration.model, candidate, gguf),
            "the whole model was placed in system RAM; generation will be slow")


# --------------------------------------------------------------------------- #
# The runtime itself
# --------------------------------------------------------------------------- #

RESIDENCY_KEY = "llm:llama.cpp"


@dataclass
class Report:
    """What the last load did, for the status line and the residency panel."""

    placement: mc_llm_context.Placement | None = None
    estimate: mc_llm_context.Estimate | None = None
    notes: tuple[str, ...] = ()
    observed_bytes: int = 0
    started_at: float = 0.0
    model: str = ""
    fits: bool = True


class Runtime:
    """One managed llama-server, placed by the broker and reclaimable by it."""

    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._signature: tuple | None = None
        self._placement: mc_llm_context.Placement | None = None
        self.report = Report()

    # -- lifecycle -------------------------------------------------------- #

    def client(self, needs_vision: bool = False):
        """A client for a ready server, started or restarted as placement requires.

        Mirrors ``prompt_master.inference.service.InferenceService.client`` --
        the same validation, the same signature comparison, the same readiness
        wait -- and differs in exactly one way, which is the reason it is not
        simply called: the context size and offload this starts the server with
        come from :func:`negotiate` rather than from the state file, because
        the state file does not know what else is on the card.
        """
        from prompt_master.inference.device_detection import CPU_DEVICE, NO_OFFLOAD
        from prompt_master.inference.llama_client import LlamaClient
        from prompt_master.inference.service import CPU_READY_TIMEOUT, GPU_READY_TIMEOUT

        with self._lock:
            configuration = config()
            if not configuration.configured:
                raise NotConfigured(
                    "No local model is configured yet. Choose a GGUF and a llama.cpp runtime "
                    "in LLM Studio’s Setup mode."
                )
            for label, path in (("llama-server", configuration.runtime),
                                ("model", configuration.model),
                                ("vision projector", configuration.mmproj)):
                if path is not None and not Path(path).is_file():
                    raise RuntimeError(f"Configured {label} is missing: {path}")
            if needs_vision and configuration.mmproj is None:
                raise RuntimeError(
                    "This request carries an image, and the model running has no vision "
                    "projector. Choose one in LLM Studio’s Setup mode, or send the "
                    "request without the image; text-only fallback is disabled."
                )

            negotiated = negotiate(configuration)
            placement = negotiated.placement
            signature = (configuration.runtime, configuration.model, configuration.mmproj,
                         configuration.gpu_index, configuration.device,
                         placement.context, placement.gpu_layers)

            if self._running and signature == self._signature:
                mc_broker.declare(mc_broker.FAMILY_LLM, RESIDENCY_KEY,
                                  self._label(configuration),
                                  self.report.observed_bytes or
                                  (negotiated.estimate.resident_bytes if negotiated.estimate else 0),
                                  rank=mc_broker.RANK_HOT)
                return LlamaClient(f"http://127.0.0.1:{self._process.port}", self._process.api_key)

            self._stop_locked("making way for a new placement")

            before = mc_broker.free_vram_bytes()
            layers = (NO_OFFLOAD if placement.gpu_layers == mc_llm_context.NO_LAYERS
                      else "all" if placement.gpu_layers == mc_llm_context.ALL_LAYERS
                      else str(placement.gpu_layers))
            paths = mc_llm_paths.app_paths()
            paths.logs.mkdir(parents=True, exist_ok=True)

            process = self._new_process()
            process.start(configuration.runtime, configuration.model, configuration.mmproj,
                          configuration.gpu_index, configuration.device, placement.context,
                          paths.logs / "llama-server.log", gpu_layers=layers)
            from_system_ram = (configuration.device.casefold() == CPU_DEVICE
                               or layers == NO_OFFLOAD)
            try:
                process.wait_ready(CPU_READY_TIMEOUT if from_system_ram else GPU_READY_TIMEOUT)
            except Exception:
                process.stop()
                raise

            self._process, self._signature, self._placement = process, signature, placement
            observed = max(before - mc_broker.free_vram_bytes(), 0) if before > 0 else 0
            self._record(configuration, negotiated, observed)
            return LlamaClient(f"http://127.0.0.1:{process.port}", process.api_key)

    def _new_process(self):
        from prompt_master.inference.llama_process import LlamaProcess

        return LlamaProcess()

    @property
    def _running(self) -> bool:
        return self._process is not None and self._process.running

    def running(self) -> bool:
        with self._lock:
            return self._running

    def _record(self, configuration: Config, negotiated: Negotiation, observed: int) -> None:
        self.report = Report(
            placement=negotiated.placement, estimate=negotiated.estimate,
            notes=negotiated.notes, observed_bytes=observed, started_at=time.time(),
            model=Path(configuration.model).name if configuration.model else "",
            fits=negotiated.fits,
        )
        for text in negotiated.notes:
            mc_broker.note(mc_broker.FAMILY_LLM, text)

        if negotiated.placement.on_gpu and observed > 0:
            mc_broker.declare(mc_broker.FAMILY_LLM, RESIDENCY_KEY, self._label(configuration),
                              observed, rank=mc_broker.RANK_HOT)
            mc_llm_context.record_observation(configuration.model, negotiated.placement, observed)
        elif negotiated.placement.on_gpu:
            mc_broker.declare(mc_broker.FAMILY_LLM, RESIDENCY_KEY, self._label(configuration),
                              negotiated.estimate.resident_bytes, rank=mc_broker.RANK_HOT)
        else:
            mc_broker.retire(RESIDENCY_KEY)

        logger.info(
            "Model Chain: llama-server ready — %s, %s token context, %.1f GB VRAM%s",
            negotiated.placement.describe(),
            f"{negotiated.placement.context:,}",
            observed / _GB,
            "" if not negotiated.notes else f" ({'; '.join(negotiated.notes)})",
        )

    def _label(self, configuration: Config) -> str:
        name = configuration.quantization or (
            Path(configuration.model).stem if configuration.model else "the LLM")
        return f"the LLM ({name})"

    # -- the broker's reclaimer ------------------------------------------- #

    def release(self, needed_bytes: int, reason: str = "") -> int:
        """Give VRAM back. Registered with the broker as the LLM family's reclaimer.

        ``needed_bytes`` is honoured in spirit rather than to the byte: a
        process either holds its allocation or it does not, and there is no
        partial surrender short of restarting somewhere else. So a request for
        any amount releases the whole of it, which is the only granularity the
        mechanism has -- and is why :func:`negotiate` works hard not to have to
        ask for image VRAM in the first place.
        """
        # Timed rather than blocking, and that is a deliberate asymmetry. The
        # two reclaim paths take their locks in opposite orders -- an LLM load
        # holds this lock and asks mc_memory for room, an image pass holds
        # nothing and asks this for room, but mc_memory's own lock sits inside
        # the first path -- so a blocking acquire here is a cycle waiting for
        # the one moment the workload lock fails to keep the two apart. Giving
        # up costs an eviction that did not happen, which the caller already
        # handles: it is reported as a shortfall and the driver spills. A
        # deadlock would cost the whole WebUI.
        if not self._lock.acquire(timeout=RELEASE_LOCK_TIMEOUT):
            logger.warning("Model Chain: the LLM runtime was busy and could not release VRAM "
                           "for %s", reason or "another workload")
            return 0
        try:
            if not self._running:
                mc_broker.retire(RESIDENCY_KEY)
                return 0
            if self._placement is not None and not self._placement.on_gpu:
                return 0  # already in system RAM; there is nothing on the card to give

            before = mc_broker.free_vram_bytes()
            if _release_mode() == RELEASE_SYSTEM_RAM:
                freed = self._restart_in_system_ram(before, reason)
                if freed:
                    return freed
            self._stop_locked(reason or "another workload needed the VRAM")
            freed = max(mc_broker.free_vram_bytes() - before, 0)
            mc_broker.note(
                mc_broker.FAMILY_LLM,
                f"stopped llama-server for {reason or 'another workload'}; the weights stay warm "
                f"in the system page cache, so restarting it re-reads from RAM, not disk"
            )
            return freed or self.report.observed_bytes
        finally:
            self._lock.release()

    def _restart_in_system_ram(self, before: int, reason: str) -> int:
        """Move the model off the card without losing the loaded server."""
        from prompt_master.inference.device_detection import NO_OFFLOAD
        from prompt_master.inference.service import CPU_READY_TIMEOUT

        configuration = config()
        if not configuration.configured:
            return 0
        try:
            self._stop_locked("moving the LLM to system RAM")
            process = self._new_process()
            paths = mc_llm_paths.app_paths()
            paths.logs.mkdir(parents=True, exist_ok=True)
            placement = (self._placement or mc_llm_context.Placement()).with_layers(
                mc_llm_context.NO_LAYERS)
            process.start(configuration.runtime, configuration.model, configuration.mmproj,
                          configuration.gpu_index, configuration.device, placement.context,
                          paths.logs / "llama-server.log", gpu_layers=NO_OFFLOAD)
            process.wait_ready(CPU_READY_TIMEOUT)
        except Exception:
            logger.warning("Model Chain: could not move the LLM to system RAM; stopping it instead",
                           exc_info=True)
            return 0

        self._process = process
        self._placement = placement
        self._signature = (configuration.runtime, configuration.model, configuration.mmproj,
                           configuration.gpu_index, configuration.device,
                           placement.context, placement.gpu_layers)
        mc_broker.retire(RESIDENCY_KEY)
        freed = max(mc_broker.free_vram_bytes() - before, 0)
        mc_broker.note(mc_broker.FAMILY_LLM,
                       f"moved the LLM to system RAM for {reason or 'another workload'}; it stays "
                       f"loaded and answering, more slowly")
        return freed or self.report.observed_bytes

    def resident_bytes(self) -> int:
        with self._lock:
            if not self._running or (self._placement is not None and not self._placement.on_gpu):
                return 0
            return self.report.observed_bytes or (
                self.report.estimate.resident_bytes if self.report.estimate else 0)

    def describe(self) -> str:
        return self._label(config())

    def stop(self) -> None:
        with self._lock:
            self._stop_locked("stop requested")

    def _stop_locked(self, reason: str) -> None:
        process, self._process = self._process, None
        self._signature, self._placement = None, None
        mc_broker.retire(RESIDENCY_KEY)
        if process is None:
            return
        try:
            process.stop()
        except Exception:
            logger.warning("Model Chain: failed to stop llama-server (%s)", reason, exc_info=True)

    # -- status ----------------------------------------------------------- #

    def status(self) -> dict:
        """Everything the panel needs about the runtime, and nothing that costs a load."""
        with self._lock:
            configuration = config()
            return {
                "configured": configuration.configured,
                "has_runtime": configuration.runtime is not None,
                "has_model": configuration.model is not None,
                "running": self._running,
                "model": Path(configuration.model).name if configuration.model else "",
                "quantization": configuration.quantization,
                "device": configuration.device_name or configuration.device,
                "mode": configuration.mode,
                "sees": configuration.sees,
                "placement": self._placement,
                "report": self.report,
                "resident_bytes": self.resident_bytes(),
            }


runtime = Runtime()
"""The single managed server. One process per WebUI, as the standalone app had
one per window: llama.cpp serialises requests within a process anyway, and two
would double the VRAM for no concurrency the broker would allow to be used."""

mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, runtime)


def shutdown() -> None:
    """Stop the server. Called from ``on_script_unloaded``."""
    runtime.stop()
