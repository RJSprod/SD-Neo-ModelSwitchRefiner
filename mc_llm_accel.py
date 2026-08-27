"""Two settings, because they are two decisions: how tokens are produced, and
who owns the card while they are.

The obvious way to ship speculative decoding is as one switch called something
like "Lightning", meaning *use the fast decoder and empty the card for it*. It
is the wrong shape, and it is wrong in a way that costs users VRAM they did not
have to spend. Those are separate facts:

* **Acceleration** is a decoding mechanism. Multi-token prediction and DFlash2
  produce the same tokens as ordinary decoding and produce more of them per
  step; neither is a quality setting and neither is a memory setting.
* **Memory priority** is a statement about ownership. Cooperative is what this
  extension has always done -- the language model lives in the VRAM the image
  side is not using and never asks for more. LLM priority is a user saying, in
  as many words, that on *this card* the language model may take room back.

Fold them together and a machine whose target and draft already fit has to
evict a checkpoint to get the decoder it could have had for nothing. So they
are two controls, the presets are a mapping over them, and the advanced
combination the presets do not offer -- DFlash2 with cooperative memory -- is
the one that matters most:

    accelerator      auto | none | mtp | dflash2
    memory priority  cooperative | llm_priority

    Normal              auto      + cooperative
    Fast LLM            mtp       + llm_priority
    Lightning Fast LLM  dflash2   + llm_priority

Where the launch contract lives, and why here
---------------------------------------------
The registry may not carry command lines -- a checked-in JSON file says *the
draft must be wholly resident on the same card as the target*, and this module
is what turns that into ``--spec-draft-model`` and the flags beside it. Every
one of them is gated on the runtime advertising it, because a flag passed to a
build that does not know it is not a slower server, it is a server that exits
at startup.

What this module will not do
----------------------------
It has no opinion about temperature or top_p, does not read a placement, does
not name a GPU index and does not stop anything. Acceleration is an execution
mechanism: :mod:`mc_llm_runtime` decides where a model goes and :mod:`mc_broker`
decides what may be moved, exactly as they did before this file existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3


# --------------------------------------------------------------------------- #
# The two axes
# --------------------------------------------------------------------------- #

ACCEL_AUTO = "auto"
ACCEL_NONE = "none"
ACCEL_MTP = "mtp"
ACCEL_DFLASH2 = "dflash2"

ACCELERATORS = (
    (ACCEL_AUTO, "Auto — the fastest mechanism this backbone and runtime both have"),
    (ACCEL_NONE, "None — ordinary decoding"),
    (ACCEL_MTP, "MTP — the backbone's own multi-token prediction heads"),
    (ACCEL_DFLASH2, "DFlash2 — the separately installed speculative draft model"),
)
"""``(value, label)`` pairs, the shape every option table in this extension has.

``auto`` is a *policy* and the other three are requests, which is the whole of
the difference in how they fail: auto steps down through the mechanisms it can
prove are available and names the one it used, and a request that cannot be
honoured is reported rather than quietly replaced. See :data:`Plan.refusal`.
"""

PRIORITY_COOPERATIVE = "cooperative"
PRIORITY_LLM = "llm_priority"

PRIORITIES = (
    (PRIORITY_COOPERATIVE, "Cooperative — use the VRAM that is already free"),
    (PRIORITY_LLM, "LLM priority — this card's image residency may be released for it"),
)
"""What the language model is allowed to ask of the card it is placed on.

Cooperative is the default and is the behaviour every version of this extension
has had: :func:`mc_broker._victim_order` returns nothing for an LLM request, so
the language model is sized against what is spare and shrinks itself when there
is little. LLM priority does not reverse that rule in general -- it opens one
door, on one card, because a user opened it.
"""


# --------------------------------------------------------------------------- #
# The presets, which are a mapping and not a third setting
# --------------------------------------------------------------------------- #

PRESET_NORMAL = "normal"
PRESET_FAST = "fast"
PRESET_LIGHTNING = "lightning"
PRESET_CUSTOM = "custom"

PRESETS = (
    (PRESET_NORMAL, "Normal"),
    (PRESET_FAST, "Fast LLM"),
    (PRESET_LIGHTNING, "Lightning Fast LLM"),
)
"""The three a user picks between. :data:`PRESET_CUSTOM` is not among them.

It is what the panel *reports* when the advanced controls hold a combination no
preset names, which is a state to be able to display rather than a state to be
able to choose -- picking "Custom" from a menu would mean nothing, because the
values it would apply are the ones already in force.
"""

PRESET_AXES: dict[str, tuple[str, str]] = {
    PRESET_NORMAL: (ACCEL_AUTO, PRIORITY_COOPERATIVE),
    PRESET_FAST: (ACCEL_MTP, PRIORITY_LLM),
    PRESET_LIGHTNING: (ACCEL_DFLASH2, PRIORITY_LLM),
}
"""What each preset means in the two axes. The single source of that mapping.

Fast asks for MTP rather than for auto, and the difference is what happens when
the backbone has no MTP heads: auto would find DFlash2 and use it, which is not
what somebody who chose the middle preset asked for. MTP is embedded and free,
so Fast is "use what the model already has, and give it room"; when the model
has none, Fast is ordinary decoding with room, and the status line says so.
"""

PRESET_DETAIL = {
    PRESET_NORMAL: ("Ordinary decoding unless the backbone and runtime both offer something "
                    "faster. The language model uses the VRAM that is free and never asks "
                    "the image side for any."),
    PRESET_FAST: ("The backbone's own multi-token heads, where it has them, and permission "
                  "to release image residency on this card if the model needs the room."),
    PRESET_LIGHTNING: ("The DFlash2 draft model, which must be wholly resident on the same "
                       "card as the backbone, and permission to release image residency on "
                       "that card to get it there. Never a partial offload."),
}
"""One sentence per preset for the panel, saying what it will and will not do.

The third sentence of Lightning is the one that had to be written down: a run
that could not fit the whole plan and quietly put half of it in system RAM
would still be labelled Lightning, and would be slower than Normal.
"""


def preset_axes(preset: str) -> tuple[str, str]:
    """``(accelerator, memory priority)`` for ``preset``, Normal's for anything else."""
    return PRESET_AXES.get(_named(preset, PRESETS, PRESET_NORMAL),
                           PRESET_AXES[PRESET_NORMAL])


def preset_for(accelerator: str, memory_priority: str) -> str:
    """Which preset names this pair of axes, or :data:`PRESET_CUSTOM`.

    The reverse of :data:`PRESET_AXES`, computed rather than written down a
    second time -- two tables that had to agree would eventually not.
    """
    wanted = (accelerator, memory_priority)
    for name, axes in PRESET_AXES.items():
        if axes == wanted:
            return name
    return PRESET_CUSTOM


# --------------------------------------------------------------------------- #
# What the installation is set to
# --------------------------------------------------------------------------- #

PREF_PRESET = "llm_performance_preset"
PREF_ACCELERATOR = "llm_accelerator"
PREF_MEMORY_PRIORITY = "llm_memory_priority"

PREFERENCES = (PREF_PRESET, PREF_ACCELERATOR, PREF_MEMORY_PRIORITY)
"""The three keys this module owns in the preferences file.

In that file rather than on the WebUI's Settings page, unlike the context and
cache settings beside them, and for the reason roles exist: a Creative Writer
pinned to a 3090 and a Spatial Composer pinned to a 5090 have different amounts
of room and may reasonably want different answers here. A Settings page holds
one value for the installation; this is drawn in Setup, beside the role picker
that decides whose setting is being edited.
"""

DEFAULTS = {
    PREF_PRESET: PRESET_NORMAL,
    PREF_ACCELERATOR: ACCEL_AUTO,
    PREF_MEMORY_PRIORITY: PRIORITY_COOPERATIVE,
}
"""Defaults that reproduce the behaviour of every build before this one.

Cooperative is the important one. An upgrade must not begin releasing image
residency because a feature was added, so the value an installation that has
never opened this panel resolves to is exactly the rule it already had.
"""


@dataclass(frozen=True)
class Settings:
    """What one configuration -- the installation's, or one role's -- asks for.

    ``preset`` is derived rather than stored-and-trusted: the advanced controls
    are authoritative, so what the panel shows as the chosen preset is whatever
    those two values happen to name. A stored preset that disagrees with them
    is a stale label, and labels do not get to decide what runs.
    """

    accelerator: str = ACCEL_AUTO
    memory_priority: str = PRIORITY_COOPERATIVE

    @property
    def preset(self) -> str:
        return preset_for(self.accelerator, self.memory_priority)

    @property
    def cooperative(self) -> bool:
        return self.memory_priority != PRIORITY_LLM

    @property
    def forced(self) -> bool:
        """Whether a specific mechanism was asked for rather than left to auto.

        The flag the whole failure contract turns on: a forced request that
        cannot be met is reported, and an automatic one steps down quietly to
        the next mechanism and names it.
        """
        return self.accelerator != ACCEL_AUTO

    def describe(self) -> str:
        return f"{label_for(ACCELERATORS, self.accelerator)} · " \
               f"{label_for(PRIORITIES, self.memory_priority)}"


def settings(role: str = "") -> Settings:
    """What ``role`` is configured to ask for. Never raises.

    Layered the way :func:`mc_llm_runtime.config` layers everything else: the
    defaults above, the preferences file, then the role's own overrides where
    it has been split. A role that has not been split follows the installation,
    which is inheritance rather than a copy for the reason :mod:`mc_llm_roles`
    gives at length.
    """
    try:
        import mc_llm_roles
        import mc_llm_state

        stored = mc_llm_state.preferences()
        if mc_llm_roles.named(role):
            stored = mc_llm_roles.layered(role, stored, None, stored,
                                          keys=mc_llm_roles.PREFS_FIELDS)
    except Exception:
        logger.debug("Model Chain: could not read the performance preferences",
                     exc_info=True)
        return Settings()
    return Settings(
        accelerator=_named(stored.get(PREF_ACCELERATOR), ACCELERATORS, ACCEL_AUTO),
        memory_priority=_named(stored.get(PREF_MEMORY_PRIORITY), PRIORITIES,
                               PRIORITY_COOPERATIVE),
    )


def remember(role: str = "", preset: str | None = None, accelerator: str | None = None,
             memory_priority: str | None = None) -> Settings:
    """Record a choice and return what is now in force.

    A preset is expanded here rather than stored as a third setting, so there
    is exactly one place the mapping lives and no way for the label and the
    axes to drift apart. Passing a preset *and* an axis is not a conflict to
    resolve: the axis wins, because that is the direction a user moves in when
    they open the advanced controls after choosing a preset.
    """
    if preset is not None:
        chosen = _named(preset, PRESETS, PRESET_NORMAL)
        wanted_accelerator, wanted_priority = preset_axes(chosen)
    else:
        current = settings(role)
        wanted_accelerator, wanted_priority = current.accelerator, current.memory_priority
    if accelerator is not None:
        wanted_accelerator = _named(accelerator, ACCELERATORS, wanted_accelerator)
    if memory_priority is not None:
        wanted_priority = _named(memory_priority, PRIORITIES, wanted_priority)

    values = {
        PREF_ACCELERATOR: wanted_accelerator,
        PREF_MEMORY_PRIORITY: wanted_priority,
        PREF_PRESET: preset_for(wanted_accelerator, wanted_priority),
    }
    try:
        import mc_llm_state

        mc_llm_state.remember_for(role, **values)
    except Exception:
        logger.warning("Model Chain: could not record the performance settings",
                       exc_info=True)
    return Settings(accelerator=wanted_accelerator, memory_priority=wanted_priority)


def follows_installation(role: str) -> bool:
    """Whether ``role`` has performance settings of its own.

    False for the installation itself, which has nothing to follow, and for a
    role nobody has split -- and those two answers are deliberately the same
    one, because both mean "read the installation's values".
    """
    try:
        import mc_llm_roles
        import mc_llm_state

        chosen = mc_llm_roles.named(role)
        if not chosen:
            return False
        stored = mc_llm_state.preferences()
        return not any(key in mc_llm_roles.overrides(chosen, None, stored)
                       for key in PREFERENCES)
    except Exception:
        return False


def _named(raw, table, default: str) -> str:
    """``raw`` as one of ``table``'s values, accepting either half of the pair.

    The same resolver :mod:`mc_broker` and :mod:`mc_llm_state` both carry, for
    the same reason: what a Gradio radio stores is the string it displayed.
    """
    text = str(raw or "").strip()
    if not text:
        return default
    folded = text.casefold()
    for value, label in table:
        if folded in (value.casefold(), label.casefold()):
            return value
    return default


def label_for(table, value: str) -> str:
    """``table``'s display text for ``value``, or the value when it has none."""
    for candidate, label in table:
        if candidate == value:
            return label
    return str(value or "")


def short_label(accelerator: str) -> str:
    """One word for a status line: ``DFlash2``, ``MTP``, ``ordinary decoding``."""
    return {ACCEL_DFLASH2: "DFlash2", ACCEL_MTP: "MTP",
            ACCEL_NONE: "ordinary decoding", ACCEL_AUTO: "auto"}.get(accelerator,
                                                                     str(accelerator or ""))


# --------------------------------------------------------------------------- #
# The launch contract
# --------------------------------------------------------------------------- #

SPEC_MODEL_FLAG = "--spec-draft-model"
SPEC_TYPE_FLAG = "--spec-type"
SPEC_MAX_FLAG = "--spec-draft-n-max"
SPEC_MIN_FLAG = "--spec-draft-n-min"
SPEC_P_MIN_FLAG = "--spec-draft-p-min"
SPEC_TYPE_K_FLAG = "--spec-draft-type-k"
SPEC_TYPE_V_FLAG = "--spec-draft-type-v"
DRAFT_LAYERS_FLAG = "--n-gpu-layers-draft"

SPEC_TYPE_DFLASH = "draft-dflash"
SPEC_TYPE_MTP = "mtp"
"""The publisher's tested start contract, one constant per option.

Written out rather than assembled from a string in the registry, because
section 15 of the design intent forbids executable command strings in a data
file and because these are the flags a reviewer has to be able to find. The
*values* beside them -- how many tokens to draft, what the draft's cache types
are -- do come from the registry, where they are numbers and cache-type names
that have already been validated.
"""


@dataclass(frozen=True)
class Plan:
    """What will actually accelerate this request, and what it costs.

    The distinction between :attr:`requested` and :attr:`accelerator` is the
    whole of the failure contract. They differ only where ``requested`` was
    ``auto``: an automatic plan steps down to what it can prove is available
    and records which mechanism that was, and a *forced* plan that cannot be
    met does not step down at all -- it comes back with :attr:`refusal` set and
    nothing started.
    """

    requested: str = ACCEL_AUTO
    accelerator: str = ACCEL_NONE
    memory_priority: str = PRIORITY_COOPERATIVE
    runtime_family: str = ""
    """Which llama.cpp family this plan needs, or ``""`` for the ordinary one."""
    runtime: Path | None = None
    """The executable to start, when it is not the configured one."""
    draft: Path | None = None
    draft_bytes: int = 0
    """What the draft's weights, cache and graph are expected to take, together."""
    required_bytes: int = 0
    """The complete plan: target, its cache and state, the draft, the projector
    allowance when this request needs vision, compute, and the safety margin."""
    spendable_bytes: int = 0
    reclaimed_bytes: int = 0
    """VRAM an explicit LLM-priority request released on this card. Zero on
    every cooperative plan, always, and that is worth asserting in a test."""
    flags: tuple[str, ...] = ()
    refusal: str = ""
    """Why a forced request could not be honoured. Empty on every plan that
    can run, including one that stepped down from ``auto``."""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def forced(self) -> bool:
        return self.requested != ACCEL_AUTO

    @property
    def refused(self) -> bool:
        return bool(self.refusal)

    @property
    def speculative(self) -> bool:
        """Whether a second model has to be resident for this plan."""
        return self.accelerator == ACCEL_DFLASH2

    @property
    def identity(self) -> tuple:
        """What a warm server would have to be restarted to change.

        Section 16 asks the warm identity to distinguish the runtime family,
        the accelerator, and whether a draft model is loaded and where. Those
        are exactly these four: the mechanism, the executable, the draft file
        and the family it belongs to. The draft's *placement* is not among them
        because the contract has only one -- wholly resident on the target's
        card -- so a plan that exists at all has it.
        """
        return (self.accelerator, str(self.runtime or ""), str(self.draft or ""),
                self.runtime_family)

    @property
    def token(self) -> str:
        """This plan as one key-safe word, for keying a measured rate by.

        Empty for ordinary decoding, so that every rate this machine has
        already learned keeps the key it was written under. Section 17 is
        explicit that a DFlash rate and an ordinary one must not be averaged
        into one estimate, and separate keys are how that is arranged.
        """
        return "" if self.accelerator in (ACCEL_NONE, ACCEL_AUTO) else str(self.accelerator)

    def describe(self) -> str:
        """One clause for the status line, naming the mechanism that ran."""
        if self.accelerator == ACCEL_DFLASH2:
            return "DFlash2 · target + draft fully on GPU"
        if self.accelerator == ACCEL_MTP:
            return "MTP · the backbone's own draft heads"
        return "ordinary decoding"


def dflash2_flags(speculator, draft: Path, *, supports, layers_flag: bool = True,
                  flash_attention: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The publisher's DFlash2 start contract, minus anything this build lacks.

    ``supports`` is a predicate over long option names -- in practice
    :func:`mc_llm_runtime.runtime_supports` bound to the runtime being started
    -- and every optional flag is asked about before it is added. The two that
    are not optional are the draft model and the speculative type: without
    either of them the server is not running DFlash2 at all, so a build that
    does not advertise them produces no flags rather than a partial contract
    that would start an ordinary server under a Lightning label.

    ``flash_attention`` is passed in rather than decided here because its
    spelling changed upstream from a switch to a three-state option and
    :mod:`mc_llm_runtime` already owns that question for the ordinary path.
    """
    if not (supports(SPEC_MODEL_FLAG) and supports(SPEC_TYPE_FLAG)):
        return ()

    flags = [SPEC_MODEL_FLAG, str(draft), SPEC_TYPE_FLAG, SPEC_TYPE_DFLASH]
    if speculator.draft_tokens and supports(SPEC_MAX_FLAG):
        flags += [SPEC_MAX_FLAG, str(int(speculator.draft_tokens))]
    if supports(SPEC_MIN_FLAG):
        flags += [SPEC_MIN_FLAG, str(int(speculator.draft_min_tokens))]
    if supports(SPEC_P_MIN_FLAG):
        flags += [SPEC_P_MIN_FLAG, _decimal(speculator.draft_p_min)]
    # The draft is required to be wholly resident, so the count is "all of it"
    # in llama.cpp's own spelling of that -- a number larger than any draft has.
    if layers_flag and supports(DRAFT_LAYERS_FLAG):
        flags += [DRAFT_LAYERS_FLAG, str(EVERY_DRAFT_LAYER)]
    if speculator.draft_kv_type_k and supports(SPEC_TYPE_K_FLAG):
        flags += [SPEC_TYPE_K_FLAG, str(speculator.draft_kv_type_k)]
    if speculator.draft_kv_type_v and supports(SPEC_TYPE_V_FLAG):
        flags += [SPEC_TYPE_V_FLAG, str(speculator.draft_kv_type_v)]
    flags.extend(flash_attention)
    return tuple(flags)


def mtp_flags(multitoken, *, supports) -> tuple[str, ...]:
    """What asks a build to use a backbone's own multi-token heads.

    Nothing at all on a build that does not advertise ``--spec-type``, which is
    the honest answer rather than a failure: the heads are in the GGUF either
    way, and a runtime that cannot be told to use them decodes ordinarily. The
    caller reports which mechanism ran, so this returning empty is visible.

    No memory term goes with it. The heads were downloaded with the weights and
    are resident with the weights; unlike a draft model there is no second file
    and nothing extra to find room for.
    """
    if multitoken is None or not multitoken.embedded:
        return ()
    if not supports(SPEC_TYPE_FLAG):
        return ()
    flags = [SPEC_TYPE_FLAG, SPEC_TYPE_MTP]
    if multitoken.draft_tokens and supports(SPEC_MAX_FLAG):
        flags += [SPEC_MAX_FLAG, str(int(multitoken.draft_tokens))]
    return tuple(flags)


EVERY_DRAFT_LAYER = 999
"""What to ask for when a draft's own layer count has not been read.

The same trick and the same reason as :data:`mc_llm_runtime.EVERY_LAYER`: a
count above the model's own is clamped by llama.cpp, and one below it is a
draft partly in system RAM under a label that promises otherwise.
"""


def _decimal(value: float) -> str:
    """A float as llama.cpp will parse it, without exponent notation."""
    return f"{float(value):g}"


# --------------------------------------------------------------------------- #
# Sentences for the failures the contract names
# --------------------------------------------------------------------------- #


def no_runtime(label: str) -> str:
    return (f"DFlash2 needs a separately installed llama.cpp build, and this installation "
            f"does not have one yet. Install the DFlash2 runtime in Setup, then choose "
            f"{label} again. Nothing about the model you are running has changed.")


def no_sidecar(label: str) -> str:
    return (f"DFlash2 needs {label}'s draft model, which has not been downloaded. Install "
            f"it in Setup — it is about 3.9 GB and is kept beside the weights. Nothing "
            f"about the model you are running has changed.")


def not_validated(family: str) -> str:
    return (f"The DFlash2 runtime installed here ({family}) has not passed the text smoke "
            f"test, so it is not offered. Re-run its verification in Setup; until it "
            f"passes, this backbone decodes the way it always has.")


def vision_not_validated(family: str) -> str:
    return (f"DFlash2 vision is not validated for this runtime ({family}). The text smoke "
            f"test passed and the image one has not, and they are recorded separately on "
            f"purpose — send this request without an image, or choose another performance "
            f"mode for it.")


def does_not_fit(required: int, spendable: int, card: str, reclaimed: int = 0) -> str:
    """The fit refusal, with both numbers in it and no promises about the other card."""
    said = [
        "DFlash2 requires the backbone and its draft model to be wholly resident on one "
        f"card, and they do not fit on {card}.",
        f"Required estimate: {required / _GB:.1f} GB.",
        f"Spendable now: {spendable / _GB:.1f} GB.",
    ]
    said.append(f"{reclaimed / _GB:.1f} GB of image VRAM was released on this card and it is "
                f"still short." if reclaimed else "No image VRAM was released.")
    said.append("Choose Lightning Fast LLM to let it release image residency on this card, "
                "free VRAM yourself, reduce the context, or use a smaller quantization."
                if not reclaimed else
                "Free VRAM yourself, reduce the context, or use a smaller quantization.")
    return " ".join(said)
