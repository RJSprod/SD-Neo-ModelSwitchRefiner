"""Two settings, because they are two decisions: how tokens are produced, and
who owns the card while they are.

The obvious way to ship a "go faster" control is as one switch meaning *use the
fast decoder and empty the card for it*. It is the wrong shape, and it is wrong
in a way that costs users VRAM they did not have to spend. Those are separate
facts:

* **Acceleration** is a decoding mechanism. Multi-token prediction produces the
  same tokens as ordinary decoding and produces more of them per step; it is
  not a quality setting and not a memory setting.
* **Memory priority** is a statement about ownership. Cooperative is what this
  extension has always done -- the language model lives in the VRAM the image
  side is not using and never asks for more. LLM priority is a user saying, in
  as many words, that on *this card* the language model may take room back.

So they are two controls and the presets are a mapping over them:

    accelerator      auto | none | mtp
    memory priority  cooperative | llm_priority

    Normal    auto + cooperative
    Fast LLM  mtp  + llm_priority

The accelerator that is not here
--------------------------------
There was a third, DFlash2: a separately downloaded draft model running on a
separately built llama.cpp, because it is an unmerged pull request. It was
removed at the request of the person it was built for, whose CUDA build cannot
run it -- three and a half thousand lines of runtime family, capability record,
residency planner and Setup panel for a mechanism that could not start on the
machine it was for.

The *shape* it argued for is what stayed. Acceleration and memory priority are
still two axes rather than one switch, because that was never a fact about
DFlash2: it is a fact about the difference between how a model decodes and who
owns the card, and MTP with cooperative memory is as real a combination as it
ever was.

Where the launch contract lives, and why here
---------------------------------------------
The registry may not carry command lines -- a checked-in JSON file says *this
backbone has multi-token heads*, and this module is what turns that into
``--spec-type draft-mtp``. Every flag is gated on the runtime advertising it
*and* on the runtime accepting the value, because a flag passed to a build that
does not know it is not a slower server, it is a server that exits at startup.

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

ACCELERATORS = (
    (ACCEL_AUTO, "Auto — the fastest mechanism this backbone and runtime both have"),
    (ACCEL_NONE, "None — ordinary decoding"),
    (ACCEL_MTP, "MTP — the backbone's own multi-token prediction heads"),
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
PRESET_CUSTOM = "custom"

PRESETS = (
    (PRESET_NORMAL, "Normal"),
    (PRESET_FAST, "Fast LLM"),
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
}
"""What each preset means in the two axes. The single source of that mapping.

Fast asks for MTP by name rather than for auto, and the difference is what it
promises: "use what the model already has, and give it room". A backbone with
no multi-token heads decodes ordinarily under it -- with the memory priority
the preset also carries -- and the status line says which of the two it got.
"""

PRESET_DETAIL = {
    PRESET_NORMAL: ("Ordinary decoding unless the backbone and runtime both offer something "
                    "faster. The language model uses the VRAM that is free and never asks "
                    "the image side for any."),
    PRESET_FAST: ("The backbone's own multi-token heads, where it has them, and permission "
                  "to release image residency on this card if the model needs the room."),
}
"""One sentence per preset for the panel, saying what it will and will not do.

What each of them will *not* do is the half worth writing down: Normal never
asks the image side for a byte, and Fast asks only for this card.
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
    """One word for a status line: ``MTP`` or ``ordinary decoding``."""
    return {ACCEL_MTP: "MTP", ACCEL_NONE: "ordinary decoding",
            ACCEL_AUTO: "auto"}.get(accelerator, str(accelerator or ""))


# --------------------------------------------------------------------------- #
# The launch contract
# --------------------------------------------------------------------------- #

SPEC_TYPE_FLAG = "--spec-type"
SPEC_MAX_FLAG = "--spec-draft-n-max"

SPEC_TYPE_MTP = "draft-mtp"
"""The two options MTP needs, and the value llama.cpp spells it with.

``draft-mtp``, not ``mtp``. The prefix is worth a sentence because getting it
wrong does not degrade anything: the server refuses the argument and exits
before it loads, which arrives as "the backbone would not start" and rolls the
user back to whichever model they were on. It happened.

Written out here rather than assembled from a string in the registry, because a
data file may not carry executable command lines and because these are the
flags a reviewer has to be able to find. The *number* beside them -- how many
tokens to draft per step -- does come from the registry, where it has already
been validated as a whole number.
"""


@dataclass(frozen=True)
class Plan:
    """What will actually accelerate this request.

    The distinction between :attr:`requested` and :attr:`accelerator` is the
    whole of the failure contract. They differ only where the request could not
    be met: an automatic plan steps down to what it can prove is available and
    records which mechanism that was, and a named one that cannot run says so
    in :attr:`notes` rather than pretending.
    """

    requested: str = ACCEL_AUTO
    accelerator: str = ACCEL_NONE
    memory_priority: str = PRIORITY_COOPERATIVE
    runtime: Path | None = None
    """The executable to start, when it is not the configured one. Always
    ``None`` today -- kept because the launch reads it, and a second runtime
    family is a thing this extension has had once and may have again."""
    flags: tuple[str, ...] = ()
    refusal: str = ""
    """Why a named request could not be honoured. Empty on every plan that can
    run, including one that stepped down from ``auto``."""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def forced(self) -> bool:
        return self.requested != ACCEL_AUTO

    @property
    def refused(self) -> bool:
        return bool(self.refusal)

    @property
    def identity(self) -> tuple:
        """What a warm server would have to be restarted to change.

        The mechanism and the executable running it. Both are start-time facts:
        an MTP server is started with ``--spec-type`` and a server that was not
        cannot be given it later.
        """
        return (self.accelerator, str(self.runtime or ""))

    @property
    def token(self) -> str:
        """This plan as one key-safe word, for keying a measured rate by.

        Empty for ordinary decoding, so that every rate this machine has
        already learned keeps the key it was written under. An accelerated rate
        and an ordinary one must not be averaged into one estimate, and
        separate keys are how that is arranged.
        """
        return "" if self.accelerator in (ACCEL_NONE, ACCEL_AUTO) else str(self.accelerator)

    def describe(self) -> str:
        """One clause for the status line, naming the mechanism that ran."""
        if self.accelerator == ACCEL_MTP:
            return "MTP · the backbone's own draft heads"
        return "ordinary decoding"


def mtp_flags(multitoken, *, supports, accepts) -> tuple[str, ...]:
    """What asks a build to use a backbone's own multi-token heads.

    Two questions, and needing both is the lesson of a server that exited at
    startup. ``supports`` is a predicate over long option *names* and
    ``accepts`` is one over ``--spec-type``'s *values*: every llama.cpp build
    since the speculative framework landed advertises the option, and which
    types it will take is a list that has grown release by release. A build
    asked for one it does not have prints ``unknown speculative type`` and
    exits before it loads a tensor.

    Nothing at all when either question answers no, which is the honest result
    rather than a failure: the heads are in the GGUF regardless, and a runtime
    that cannot be told to use them decodes ordinarily. The caller reports which
    mechanism ran, so this returning empty is visible.

    No memory term goes with it. The heads were downloaded with the weights and
    are resident with the weights; there is no second file to find room for.
    """
    if multitoken is None or not multitoken.embedded:
        return ()
    if not supports(SPEC_TYPE_FLAG) or not accepts(SPEC_TYPE_MTP):
        return ()
    flags = [SPEC_TYPE_FLAG, SPEC_TYPE_MTP]
    if multitoken.draft_tokens and supports(SPEC_MAX_FLAG):
        flags += [SPEC_MAX_FLAG, str(int(multitoken.draft_tokens))]
    return tuple(flags)


def no_heads(label: str) -> str:
    return (f"{label} has no multi-token prediction heads, so this runs at ordinary "
            f"decoding speed.")


def no_option(label: str) -> str:
    return (f"This llama.cpp build does not accept the multi-token prediction options, so "
            f"{label}'s own draft heads are not used. Update the runtime in Setup to get "
            f"them.")
