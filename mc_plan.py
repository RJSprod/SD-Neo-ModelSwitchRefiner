"""The active execution plan for one generation, and the VRAM budget it implies.

Why this module exists
----------------------
Everything else in the extension sizes memory against *now*: how much is free
at this instant, what the loaded checkpoint weighs, what the running server
holds. That is the right question for one workload and the wrong question for a
generation assembled from several, because the phases do not all happen at
once and the first one to ask gets an answer computed as though it were the
only one.

The observed cost of asking it that way, from a user's ``llama-server.log``
covering one session:

* 71 server starts, of which 47 died loading the model;
* the negotiated context oscillating 7168 -> 8192 -> 7168 across consecutive
  generations, because free VRAM oscillated with it, and every change of the
  placement signature is a restart and a lost prompt cache;
* five consecutive generations where all three start attempts failed with
  ``cudaMalloc failed: out of memory`` while the card reported 22.7 GB free --
  llama.cpp read the free figure, the image side took the card during the load,
  and the language model lost the race;
* 31 starts that never reached the model at all, dying at argument parsing with
  ``invalid device: CUDA0`` because no CUDA device could be enumerated on a
  card another process had filled.

None of those is a bug in the arithmetic that ran. They are all the same bug in
*when* it ran and *what it was allowed to see*.

What replaces it
----------------
A plan is built once, before the generation starts, from the features that are
actually switched on. It names the phases that will run, what each will cost at
its peak, and which of them is the largest. The largest is what the image side
is protected for; the language model is placed in what is left over and then
left alone, because a placement that does not change is a server that does not
restart.

    Creative Writer -> Spatial Composer -> Stage 1 (Krea 2) -> Stage 2 (Klein 9B)

is a plan. So is

    Stage 1 (Krea 2)

and the arithmetic below is the same for both: no feature has its own memory
policy, it has a phase, and turning it off removes the phase rather than
selecting a different policy.

Mutually exclusive phases share; real overlaps do not
-----------------------------------------------------
Stage 1 and Stage 2 never sample at the same time, so their full residencies
are never summed -- :func:`Plan.image_working_peak` takes the maximum. The
handoff between them is where they *do* briefly overlap, because the switch
deliberately keeps Stage 1's VAE and text encoder resident while Stage 2 loads
(``mc_memory._pinned_keep``). That overlap is real, so it is a phase of its
own with its own peak, and it is frequently the largest one in the plan.

What this module does not do
----------------------------
It moves nothing and starts nothing. It answers "how much" and "which phase",
and ``mc_llm_runtime`` and ``mc_memory`` do the acting. That separation is the
reason the panel can show the same numbers the placement was made from without
the act of showing them changing anything.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

KIND_PREPARATION = "preparation"
"""Runs before any image work and holds no image residency of its own."""
KIND_IMAGE = "image"
"""Samples. Holds a full model plus its activations."""
KIND_TRANSITION = "transition"
"""Between two image phases, and the only place two of them overlap."""
KIND_WARMUP = "warm-up"
"""Speculative, after the last image phase. Yields rather than forces."""

CREATIVE_WRITER = "creative_writer"
SPATIAL_COMPOSER = "spatial_composer"
DIRECT_MERGE = "direct_merge"
STAGE_1 = "stage_1"
HANDOFF = "handoff"
STAGE_2 = "stage_2"
WARM_UP = "warm_up"

LLM_PHASES = (CREATIVE_WRITER, SPATIAL_COMPOSER)
"""Phases that need llama-server. Direct BBOX merge is deliberately not one.

A plan containing only :data:`DIRECT_MERGE` reserves nothing for a Spatial
Composer, because there is no second LLM call to reserve for -- scenario D of
the design intent, and the reason the merge mode is a phase at all rather than
an absence.
"""

CAP_OFF = "off"
CAP_AUTO = "auto"
CAP_CUSTOM = "custom"

CAP_MODES = (
    (CAP_AUTO, "Auto — size the LLM from what the active plan leaves over"),
    (CAP_CUSTOM, "Custom — never let the LLM hold more than a figure you set"),
    (CAP_OFF, "Off — no persistent LLM residency; the card is the image plan's"),
)

OPT_CAP_MODE = "model_chain_llm_cap_mode"
OPT_CAP_GB = "model_chain_llm_cap_gb"
OPT_SAFETY_GB = "model_chain_plan_safety_gb"


def option(name: str, default):
    """One host setting, or ``default`` when the host is not there to ask.

    Mirrors ``mc_memory.option`` rather than importing it, because this module
    is imported by ``mc_llm_runtime`` and ``mc_memory`` imports *that* in
    places; a straight import here would close the loop at module scope.
    """
    try:
        from modules import shared

        return getattr(shared.opts, name, default)
    except Exception:
        return default


def _float_option(name: str, default: float = 0.0) -> float:
    try:
        return max(float(option(name, default) or default), 0.0)
    except (TypeError, ValueError):
        return float(default)


def cap_mode() -> str:
    """Off / Auto / Custom, defaulting to Auto and never to a spelling nobody stored."""
    recorded = str(option(OPT_CAP_MODE, CAP_AUTO) or CAP_AUTO).strip().casefold()
    return recorded if recorded in (CAP_OFF, CAP_AUTO, CAP_CUSTOM) else CAP_AUTO


def custom_cap_bytes() -> int:
    """The user's Custom ceiling in bytes, or 0 when Custom is not selected."""
    if cap_mode() != CAP_CUSTOM:
        return 0
    return int(_float_option(OPT_CAP_GB, 0.0) * _GB)


def user_safety_bytes() -> int:
    """Extra headroom the user asked the image plan to keep, on top of the global margin."""
    return int(_float_option(OPT_SAFETY_GB, 0.0) * _GB)


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Phase:
    """One step of a generation, and what it costs at its worst moment.

    ``peak_bytes`` is everything intentionally resident while this phase runs,
    which for an image phase is weights plus activations and for a transition
    is the incoming model plus whatever the switch deliberately spares. It is
    zero for a preparation phase: the writer's own memory is llama-server's,
    which is the thing being budgeted *around*, and counting it on both sides
    would reserve it twice.

    ``measured`` says whether ``peak_bytes`` came from a reading or an
    estimate, and exists so the panel can say which -- section 21's rule that a
    phase nobody has measured must not be displayed as though somebody had.
    """

    name: str
    kind: str
    label: str
    peak_bytes: int = 0
    measured: bool = False
    detail: str = ""

    @property
    def holds_image_vram(self) -> bool:
        return self.kind in (KIND_IMAGE, KIND_TRANSITION, KIND_WARMUP)

    @property
    def needs_llm(self) -> bool:
        return self.name in LLM_PHASES

    def describe(self) -> str:
        return f"{self.label} ({self.detail})" if self.detail else self.label


@dataclass(frozen=True)
class Plan:
    """The phases this generation will actually run, in the order they run.

    Built once and then read many times. It is frozen because a plan that can
    be edited after a placement was made against it is a plan that cannot be
    compared with the one the running server was placed for, and that
    comparison is the whole of :func:`boundary_moved`.
    """

    phases: tuple[Phase, ...] = ()
    width: int = 0
    height: int = 0
    built_at: float = field(default_factory=time.time)

    # -- the peak-of-plan rule -------------------------------------------- #

    def image_phases(self) -> tuple[Phase, ...]:
        return tuple(phase for phase in self.phases if phase.holds_image_vram)

    def image_working_peak(self) -> int:
        """The largest single phase, never the sum of them.

        Stage 1 and Stage 2 take over one another's arena. Summing their
        residencies describes a machine that runs both at once, which no code
        path in this extension does, and on a 24 GB card it describes one that
        cannot run the generation at all.
        """
        return max((phase.peak_bytes for phase in self.image_phases()), default=0)

    def limiting(self) -> Phase | None:
        """The phase that sets the peak, which is what the panel names.

        Ties go to the earlier phase, because the earlier one is the one the
        user will hit first and therefore the one worth naming.
        """
        best = None
        for phase in self.image_phases():
            if best is None or phase.peak_bytes > best.peak_bytes:
                best = phase
        return best if best is not None and best.peak_bytes > 0 else None

    # -- what the plan contains ------------------------------------------- #

    def has(self, name: str) -> bool:
        return any(phase.name == name for phase in self.phases)

    def phase(self, name: str) -> Phase | None:
        for candidate in self.phases:
            if candidate.name == name:
                return candidate
        return None

    @property
    def uses_llm(self) -> bool:
        return any(phase.needs_llm for phase in self.phases)

    @property
    def llm_calls(self) -> int:
        """How many LLM requests this plan makes, which is how many a warm server saves."""
        return sum(1 for phase in self.phases if phase.needs_llm)

    # -- identity ---------------------------------------------------------- #

    def identity(self) -> tuple:
        """What has to change before a placement is worth reconsidering.

        Phase names and the *class* of each peak, not the peaks themselves. A
        byte-exact comparison would call every generation a new plan, because
        an observed activation peak moves a little on every pass -- and a plan
        that is new every generation restarts llama-server every generation,
        which is precisely the behaviour this module was written to stop.

        Quarter-gigabyte classes, which is fine enough that a real change of
        model or resolution lands in a different one and coarse enough that
        measurement noise does not. Section 11: resolution moves the placement
        when it moves into "a materially different learned memory class", and
        this is that class.
        """
        step = _GB // 4
        return tuple(
            (phase.name, phase.peak_bytes // step if step else phase.peak_bytes)
            for phase in self.phases
        )

    def describe(self) -> str:
        """The plan as one line, for the panel and the log.

        ``Creative Writer -> Spatial Composer -> Stage 1 (Krea 2) -> Stage 2 (Klein 9B)``
        """
        return " -> ".join(phase.describe() for phase in self.phases) or "nothing"


# --------------------------------------------------------------------------- #
# Building a plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stage:
    """One image stage's identity, as the panel and the estimator both need it.

    ``modules`` is the VAE / text-encoder selection, which for Flux-family and
    Krea checkpoints is several gigabytes that a checkpoint file size alone
    does not see. ``multiplier`` is Stage 2's size multiplier and is 1.0 for
    Stage 1, because Stage 1 samples at the size the user asked for.
    """

    name: str = ""
    modules: object = None
    multiplier: float = 1.0
    label: str = ""

    @property
    def present(self) -> bool:
        return bool(str(self.name or "").strip())

    def shown(self) -> str:
        text = str(self.label or self.name or "").strip()
        # Forge names a checkpoint "subdir/krea2.safetensors [a1b2c3d4]". The
        # panel wants the part a person recognises, and the log wants it to
        # match -- so the directory, the hash and the extension all come off,
        # in that order, and anything that does not look like that is left
        # exactly as it is.
        for separator in ("\\", "/"):
            text = text.rsplit(separator, 1)[-1]
        if text.endswith("]") and " [" in text:
            text = text[: text.rindex(" [")].strip()
        for suffix in (".safetensors", ".ckpt", ".gguf", ".sft", ".pt"):
            if text.casefold().endswith(suffix):
                text = text[: -len(suffix)]
                break
        return text.strip()


def _pass_bytes(stage: Stage, width: int, height: int) -> int:
    """VRAM one sampling pass on ``stage`` needs: weights resident, plus activations.

    Delegated to ``mc_memory`` rather than reimplemented, so that a plan and the
    eviction that later serves it are working from one number. Zero when the
    checkpoint cannot be sized -- a plan that cannot see a model is a plan that
    reserves nothing for it, which leaves the old behaviour rather than
    inventing a reserve out of an unknown.
    """
    if not stage.present:
        return 0
    try:
        import mc_memory

        return int(mc_memory.vram_required_bytes(stage.name, stage.modules, width, height))
    except Exception:
        logger.debug("Model Chain: could not size the %s pass", stage.shown(), exc_info=True)
        return 0


def _module_bytes(stage: Stage) -> int:
    """What Stage 1's separate VAE / text-encoder files weigh, resident.

    The handoff spares exactly these and not the UNet, so this is the overlap
    the transition peak has to carry. Read off the module files because that is
    where a Flux-family or Krea checkpoint keeps them; a checkpoint with its
    encoders built in returns 0, which is correct -- there is nothing separable
    to spare, and ``mc_memory.capture_stage_1_encoders`` will find nothing to
    pin either.
    """
    if not stage.present:
        return 0
    try:
        import mc_memory

        resolved = mc_memory.resolve_modules(stage.modules)
        if resolved is None:
            resolved = mc_memory.current_modules()
        if not resolved:
            return 0
        import os

        total = 0
        for module in resolved:
            try:
                total += os.path.getsize(module)
            except OSError:
                continue
        return int(total * (1.0 + mc_memory.VRAM_MODEL_OVERHEAD_FRACTION))
    except Exception:
        logger.debug("Model Chain: could not size Stage 1's encoders", exc_info=True)
        return 0


def _stage_2_size(width: int, height: int, multiplier: float) -> tuple[int, int]:
    """The size Stage 2 actually samples at, which is not the size Stage 1 did.

    Stage 2 refines the finished pixels, so its activations scale with the
    multiplied size. A plan that budgets Stage 2 at Stage 1's resolution
    under-reserves by the square of the multiplier, and a 1.5x multiplier is
    therefore not a 50% error but a 125% one.
    """
    try:
        scale = max(float(multiplier or 1.0), 0.0) or 1.0
    except (TypeError, ValueError):
        scale = 1.0
    return int(max(width, 0) * scale), int(max(height, 0) * scale)


def build(*, width: int = 0, height: int = 0,
          stage_1: Stage | None = None, stage_2: Stage | None = None,
          creative: bool = False, spatial_compose: str = "",
          warm_up: bool = True) -> Plan:
    """Assemble the plan for one generation from the features that are switched on.

    ``spatial_compose`` is ``"smart"`` for a Spatial Composer call, ``"direct"``
    for a deterministic merge, and empty for no layout at all. Three values
    rather than a boolean because the middle one is a phase that costs nothing
    and must still appear: a user looking at the plan needs to see that their
    layout is being applied, and a plan that showed nothing between the writer
    and Stage 1 would read as though it had been dropped.

    Every phase is optional and none of them has a policy of its own. Turning
    Stage 2 off removes a phase and a transition; turning Creative Mode off
    removes a preparation phase; the arithmetic underneath is untouched.
    """
    stage_1 = stage_1 or Stage()
    phases: list[Phase] = []

    if creative:
        phases.append(Phase(CREATIVE_WRITER, KIND_PREPARATION, "Creative Writer"))

    mode = str(spatial_compose or "").strip().casefold()
    if mode == "smart":
        phases.append(Phase(SPATIAL_COMPOSER, KIND_PREPARATION, "Spatial Composer"))
    elif mode:
        phases.append(Phase(DIRECT_MERGE, KIND_PREPARATION, "Direct BBOX Merge"))

    stage_1_peak = _pass_bytes(stage_1, width, height)
    phases.append(Phase(STAGE_1, KIND_IMAGE, "Stage 1", stage_1_peak,
                        detail=stage_1.shown()))

    if stage_2 is not None and stage_2.present:
        wide, tall = _stage_2_size(width, height, stage_2.multiplier)
        stage_2_peak = _pass_bytes(stage_2, wide, tall)
        # The overlap the switch actually creates, and only that. The whole of
        # Stage 1 is not assumed to survive into Stage 2 -- it does not, the
        # UNet is evicted -- but its encoders are deliberately spared, and
        # pretending otherwise under-reserves the one moment in the whole plan
        # where two models are on the card together.
        spared = _module_bytes(stage_1)
        phases.append(Phase(HANDOFF, KIND_TRANSITION, "Handoff",
                            stage_2_peak + spared,
                            detail=("Stage 1 encoders kept" if spared else "")))
        phases.append(Phase(STAGE_2, KIND_IMAGE, "Stage 2", stage_2_peak,
                            detail=stage_2.shown()))
        if warm_up:
            # After Stage 2 the chain restores Stage 1 and warms it for the next
            # press. It is speculative and it yields (section 16), so it is a
            # phase with a real peak that is almost never the limiting one --
            # and when it *is*, the panel should say so rather than hide it.
            phases.append(Phase(WARM_UP, KIND_WARMUP, "Stage 1 warm-up",
                                stage_1_peak, detail="restored for the next press"))

    return Plan(tuple(phases), int(max(width, 0)), int(max(height, 0)))


# --------------------------------------------------------------------------- #
# Reading the rest of the generation's configuration
# --------------------------------------------------------------------------- #

MODEL_CHAIN_TITLE = "Model Chain"
"""Located by title, as ``mc_references`` locates ImageStitch.

Not by import: the two scripts are loaded from paths by the host, and one
reaching into the other's module would tie the plan to a load order neither of
them controls. A title is what the host itself indexes them by.
"""

NO_STAGE_2 = "None"
"""``ScriptModelChain``'s own spelling for "no Stage 2 checkpoint chosen".

Duplicated rather than imported for the same reason the title is: the script
lives in ``scripts/`` and is loaded by path, so importing it from here would
work in the host and not in a test, and the constant is one word.
"""

STAGE_2_ARGUMENTS = {"enabled": 0, "target": 1, "modules": 2, "size_multiplier": 15}
"""Positions of the Stage 2 controls in ``ScriptModelChain.ui``'s return list.

Positional because that is the only form ``p.script_args`` has. Fragile in
principle, and held in place by ``tests/test_plan.py``, which asserts these
indices against the live component list rather than against a copy of it -- so
reordering the panel fails a test rather than silently mis-sizing a reserve.
"""


def _script(p, title: str):
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", None) or []:
        try:
            if script.title() == title:
                return script
        except Exception:
            continue
    return None


def stage_2_from(p) -> Stage | None:
    """Stage 2 as the Model Chain panel has it set, or None when it is not armed.

    Read off ``p.script_args`` rather than passed in, because the script that
    needs it most -- Creative Mode, deciding how much room to leave before it
    calls the language model -- is a different script and has no access to the
    other's arguments. Doing it this way also makes the answer independent of
    which of the two hooks the host happens to run first, which is not
    something either script gets to choose.

    Returns None for every reason a plan should contain no Stage 2: the panel
    is switched off, no checkpoint is chosen, the extension is not installed,
    or the arguments cannot be located. Each of those means "the generation
    ends after Stage 1", and a plan that reserved for a phase that will not run
    would take that VRAM from the language model for nothing.
    """
    script = _script(p, MODEL_CHAIN_TITLE)
    if script is None:
        return None

    start, end = getattr(script, "args_from", None), getattr(script, "args_to", None)
    if start is None or end is None:
        return None
    try:
        args = list((getattr(p, "script_args", None) or [])[start:end])
    except (TypeError, KeyError):
        return None

    def at(key, default=None):
        index = STAGE_2_ARGUMENTS[key]
        return args[index] if index < len(args) else default

    if not at("enabled"):
        return None
    target = str(at("target", "") or "").strip()
    if not target:
        return None
    if target == NO_STAGE_2:
        return None
    try:
        multiplier = float(at("size_multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        multiplier = 1.0
    return Stage(name=target, modules=at("modules"), multiplier=multiplier)


CREATIVE_TITLE = "Krea Creative Mode"

CREATIVE_ENABLED = 0
"""``ScriptKreaCreative.ui`` returns the enable flag first and always."""
SPATIAL_TAIL = 3
"""...and the three Spatial controls last and always.

Its middle is a variable number of axis controls, sized by the vocabulary the
library loaded, which is why the two ends are the two that can be read without
counting -- the script's own ``_split`` says as much in the same words.
"""


def creative_from(p) -> tuple[bool, str]:
    """``(writer will run, compose mode)`` as the Creative panel has it set.

    The mirror of :func:`stage_2_from`, and it exists for the mirror reason:
    when Creative Mode is switched off, Model Chain is the script that has to
    publish the plan, and it cannot see the other panel's controls either.

    ``compose mode`` is ``"smart"``, ``"direct"`` or empty, and empty is also
    the answer when Spatial Layout is on but the canvas has no boxes on it --
    though that last case is only visible to the Creative script itself, which
    has the parsed layout and passes it in explicitly. From here the best that
    can be said is what the controls say.
    """
    script = _script(p, CREATIVE_TITLE)
    if script is None:
        return False, ""

    start, end = getattr(script, "args_from", None), getattr(script, "args_to", None)
    if start is None or end is None:
        return False, ""
    try:
        args = list((getattr(p, "script_args", None) or [])[start:end])
    except (TypeError, KeyError):
        return False, ""
    if not args or not args[CREATIVE_ENABLED]:
        return False, ""

    if len(args) < CREATIVE_ENABLED + 1 + SPATIAL_TAIL:
        # An API request that sent only the flag, or a panel that could not
        # build its axis controls. Creative Mode still runs; there is no layout.
        return True, ""
    spatial_enabled, compose = args[-SPATIAL_TAIL], args[-SPATIAL_TAIL + 1]
    if not spatial_enabled:
        return True, ""
    mode = str(compose or "").strip().casefold()
    return True, mode if mode in ("smart", "direct") else "smart"


def build_for(p, *, creative: bool | None = None,
              spatial_compose: str | None = None) -> Plan:
    """The plan for the generation ``p`` describes, assembled from every source.

    Stage 1 is the loaded checkpoint and the modules selected for it; Stage 2 is
    whatever the Model Chain panel has armed; the preparation phases are what
    the caller says they are, because only the caller knows whether it is about
    to call a writer.

    The size is the size Stage 1 *finishes* at, not the size it starts at:
    with hires fix on, ``p.width``/``p.height`` still describe the first pass
    while the activations that decide the peak -- and the image Stage 2
    receives -- are the upscaled ones.

    ``creative`` and ``spatial_compose`` default to reading the Creative panel's
    own controls off ``p``, so that either script can call this and get the same
    plan. The Creative script passes them explicitly because it alone knows
    whether the layout it just parsed actually has boxes in it -- Spatial Layout
    switched on over an empty canvas runs no Composer, and a plan that said
    otherwise would name a phase the user never sees.
    """
    if creative is None or spatial_compose is None:
        found, mode = creative_from(p)
        creative = found if creative is None else creative
        spatial_compose = mode if spatial_compose is None else spatial_compose

    width = height = 0
    try:
        import mc_arch

        width, height = mc_arch.stage1_size(p)
    except Exception:
        width = int(getattr(p, "width", 0) or 0)
        height = int(getattr(p, "height", 0) or 0)

    stage_1 = Stage()
    try:
        from modules import shared

        name = str(getattr(shared.opts, "sd_model_checkpoint", "") or "")
        if name:
            stage_1 = Stage(name=name)
    except Exception:
        logger.debug("Model Chain: could not read the Stage 1 checkpoint for the plan",
                     exc_info=True)

    return build(width=width, height=height, stage_1=stage_1,
                 stage_2=stage_2_from(p), creative=creative,
                 spatial_compose=spatial_compose)


# --------------------------------------------------------------------------- #
# The current plan
# --------------------------------------------------------------------------- #

_lock = threading.RLock()
_current: Plan | None = None
_placed_for: tuple | None = None
"""The identity of the plan the running llama-server was placed for."""


def publish(plan: Plan | None) -> Plan | None:
    """Make ``plan`` the one every budget question is answered from.

    Said at INFO when the plan is genuinely different from the last one, and
    silently when it is not. A user pressing Generate ten times with the same
    settings should see the plan once, and a user who just switched Stage 2 on
    should see it again immediately.
    """
    global _current

    with _lock:
        previous = _current
        _current = plan
        if plan is None:
            return None
        if previous is None or previous.identity() != plan.identity():
            limiting = plan.limiting()
            logger.info(
                "Model Chain: active plan — %s; image working peak %.1f GB%s",
                plan.describe(),
                plan.image_working_peak() / _GB,
                f", set by {limiting.label}" if limiting else "",
            )
        return plan


def current() -> Plan | None:
    with _lock:
        return _current


def clear() -> None:
    """Forget the plan, so a question asked outside a generation is answered honestly.

    Not the same as publishing an empty plan. No plan means "there is nothing
    to protect, use what is free", which is the correct answer for LLM Studio
    writing a prompt with no image generation behind it; an empty plan would
    mean "the image side needs nothing", and would let the language model take
    the whole card and hold it.
    """
    global _current

    with _lock:
        _current = None


def note_placement(plan: Plan | None) -> None:
    """Record which plan the server that is now running was placed for."""
    global _placed_for

    with _lock:
        _placed_for = plan.identity() if plan is not None else None


def placed_for() -> tuple | None:
    with _lock:
        return _placed_for


def boundary_moved() -> bool:
    """Whether the plan has changed enough to justify re-placing the LLM.

    This is the whole of section 11 and rule 10, and it is a *negative*
    function: nearly every time it is asked the answer is no, and no means the
    running server is left exactly as it is.

    A normal phase transition inside one generation is not a plan boundary, so
    Stage 1 finishing does not reach here. Neither does free VRAM changing,
    which is what used to drive the decision and is the reason a session's log
    shows the negotiated context stepping 7168 -> 8192 -> 7168 across
    consecutive generations, restarting the server on each step.

    A server placed with no plan in force is not re-placed merely because a
    plan has since appeared: it is already running, its placement worked, and
    the plan's job is to size the *next* one. The exception is a plan that no
    longer fits, which is a reserve miss and is handled as one.
    """
    with _lock:
        if _placed_for is None:
            return False
        plan = _current
        if plan is None:
            return False
        return plan.identity() != _placed_for


# --------------------------------------------------------------------------- #
# The budget
# --------------------------------------------------------------------------- #


def image_working_peak() -> int:
    """The active plan's peak, or 0 when no plan is in force."""
    plan = current()
    return plan.image_working_peak() if plan is not None else 0


def image_protected_bytes() -> int:
    """VRAM the active image plan is entitled to, whatever else wants it.

    The peak of the plan plus the user's own safety adjustment, and *not* plus
    the global safety margin -- ``mc_memory.vram_required_bytes`` has already
    folded that into every phase peak, and adding it again here is a gigabyte
    and a half of a card set aside twice for one activation peak.

    Zero with no plan in force, which means "protect nothing", which is right:
    outside a generation there is no image plan to protect and the language
    model may use what is free.
    """
    plan = current()
    if plan is None:
        return 0
    return plan.image_working_peak() + user_safety_bytes()


def usable_vram_bytes() -> int:
    try:
        import mc_broker

        return int(mc_broker.total_vram_bytes())
    except Exception:
        return 0


def persistent_llm_budget() -> int:
    """What the language model may hold persistently, given the active plan.

    The remainder, and only ever the remainder: rule 6. On Off it is zero, on
    Custom it is the smaller of the remainder and the user's ceiling, and on
    Auto it is the remainder alone.

    A Custom cap *above* the calculated allowance does not raise it. The
    control is documented as a way to be more conservative than the arithmetic,
    and a control that could also be less conservative than it would be a way
    to break image generation from a text box.
    """
    mode = cap_mode()
    if mode == CAP_OFF:
        return 0

    total = usable_vram_bytes()
    if total <= 0:
        # No card, or no way to ask about it. The negotiator's existing
        # behaviour -- size against what is free -- is the honest fallback, and
        # a budget of zero would wrongly force the model into system RAM.
        return -1

    allowance = max(total - image_protected_bytes(), 0)
    ceiling = custom_cap_bytes()
    if mode == CAP_CUSTOM and ceiling > 0:
        return min(allowance, ceiling)
    return allowance


def llm_reserve_bytes() -> int:
    """What to tell :func:`mc_llm_runtime.negotiate` to keep clear for the image plan.

    Expressed as a reserve rather than a budget because that is the shape the
    negotiator already takes, and because the two differ by what the image side
    is *already* holding: a checkpoint resident in fourteen gigabytes has
    already taken its share out of free VRAM, and reserving it a second time
    sizes the language model against a card that does not exist.

    The global safety margin comes off for the same reason it does in
    :func:`image_protected_bytes` -- ``negotiate`` adds it on top of whatever
    this returns, and it is inside every phase peak already.
    """
    protected = image_protected_bytes()
    if protected <= 0:
        return 0
    try:
        import mc_broker

        held = max(int(mc_broker.held_bytes(mc_broker.FAMILY_IMAGE)), 0)
        margin = max(int(mc_broker.safety_margin_bytes()), 0)
    except Exception:
        logger.debug("Model Chain: could not read the image family's residency",
                     exc_info=True)
        return max(protected, 0)
    return max(protected - held - margin, 0)


# --------------------------------------------------------------------------- #
# Reserve misses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Miss:
    """One occasion when the plan's arithmetic was wrong, and what it cost.

    Recorded rather than logged and forgotten, because section 24 asks the UI
    to show the miss and section 16 asks it to be clear that a miss is not
    normal behaviour. A number that only ever appears in a log file is a number
    the person who needs to lower their cap will never see.
    """

    phase: str = ""
    shortfall_bytes: int = 0
    llm_bytes: int = 0
    evicted: bool = False
    cap_bytes: int = 0
    suggested_bytes: int = 0
    at: float = field(default_factory=time.time)

    def describe(self) -> str:
        text = (f"{self.phase or 'an image phase'} exceeded the protected image budget "
                f"by {self.shortfall_bytes / _GB:.1f} GB")
        if self.evicted:
            text += "; llama-server was emergency-evicted"
        if self.cap_bytes > 0:
            text += f". Active LLM cap: {self.cap_bytes / _GB:.1f} GB"
        if self.suggested_bytes > 0:
            text += f". Suggested safer cap: {self.suggested_bytes / _GB:.1f} GB"
        return text + "."


_misses: list[Miss] = []
_learned_cap: int = 0
MISS_HISTORY = 8
MISS_MARGIN = int(0.5 * _GB)
"""How much *further* below the shortfall a suggested cap goes.

A cap set to exactly the shortfall is a cap that will miss again the first time
an activation peak lands a little higher, and the second miss costs another
eviction and another restart. Being wrong in this direction costs tokens per
second; being wrong in the other costs the generation.
"""


def record_miss(phase: str, shortfall_bytes: int, *, llm_bytes: int = 0,
                evicted: bool = False) -> Miss:
    """File a reserve miss and learn a safer cap from it.

    The learned figure is a floor on nothing and a *ceiling* on the next
    automatic allowance: rule 16's "do not silently promote the LLM back to the
    previous aggressive placement". It survives until the plan changes, because
    a miss on this plan says nothing about a different one.
    """
    global _learned_cap

    # With no plan in force there is no allowance to quote, and quoting the
    # whole card -- which is what the budget arithmetic returns when nothing is
    # protected -- would print "Active LLM cap: 24.0 GB" at a user whose
    # language model was holding four.
    cap = max(persistent_llm_budget(), 0) if current() is not None else 0
    held = max(int(llm_bytes), 0)
    shortfall = max(int(shortfall_bytes), 0)
    # What the language model should have been holding for the image to have
    # fitted, less a margin so the next attempt is not sized to the exact
    # figure that just failed.
    safer = max(held - shortfall - MISS_MARGIN, 0) if held else max(cap - shortfall, 0)

    miss = Miss(phase=str(phase or ""), shortfall_bytes=shortfall, llm_bytes=held,
                evicted=bool(evicted), cap_bytes=cap, suggested_bytes=safer)

    with _lock:
        _misses.append(miss)
        del _misses[:-MISS_HISTORY]
        if safer > 0 and (_learned_cap <= 0 or safer < _learned_cap):
            _learned_cap = safer

    logger.warning("Model Chain: reserve miss — %s", miss.describe())
    return miss


def misses(limit: int = MISS_HISTORY) -> list[Miss]:
    with _lock:
        return list(_misses[-int(max(limit, 0)):])


def last_miss() -> Miss | None:
    with _lock:
        return _misses[-1] if _misses else None


def learned_cap_bytes() -> int:
    """The ceiling a previous miss taught, or 0 if nothing has gone wrong yet."""
    with _lock:
        return _learned_cap


def forget_misses() -> None:
    """Clear the miss history and the learned cap. An explicit user action only."""
    global _learned_cap

    with _lock:
        del _misses[:]
        _learned_cap = 0


# --------------------------------------------------------------------------- #
# What the panel reads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Budget:
    """Everything section 21 asks the txt2img panel to show, in one object.

    Assembled here rather than in the panel so that the panel and the log
    cannot disagree about what the reserve was -- they read the same fields off
    the same snapshot, taken at one moment.
    """

    plan: Plan | None = None
    total_bytes: int = 0
    working_peak_bytes: int = 0
    limiting: Phase | None = None
    safety_bytes: int = 0
    user_safety_bytes: int = 0
    protected_bytes: int = 0
    llm_allowance_bytes: int = 0
    llm_cap_mode: str = CAP_AUTO
    llm_custom_cap_bytes: int = 0
    llm_learned_cap_bytes: int = 0

    @property
    def active(self) -> bool:
        return self.plan is not None


def budget() -> Budget:
    """One consistent reading of the whole memory contract."""
    plan = current()
    try:
        import mc_broker

        margin = int(mc_broker.safety_margin_bytes())
    except Exception:
        margin = 0
    return Budget(
        plan=plan,
        total_bytes=usable_vram_bytes(),
        working_peak_bytes=plan.image_working_peak() if plan is not None else 0,
        limiting=plan.limiting() if plan is not None else None,
        safety_bytes=margin,
        user_safety_bytes=user_safety_bytes(),
        protected_bytes=image_protected_bytes(),
        llm_allowance_bytes=max(persistent_llm_budget(), 0),
        llm_cap_mode=cap_mode(),
        llm_custom_cap_bytes=custom_cap_bytes(),
        llm_learned_cap_bytes=learned_cap_bytes(),
    )
