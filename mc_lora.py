"""Prepared model and LoRA state: what may be reused, and what must not leak.

Two problems that look unrelated and are the same problem. Both are about a
LoRA ending up attached to a model the user did not attach it to.


Reusing a prepared state
------------------------
Forge Neo's LoRA loader does not patch weights on every generation. It builds a
patched clone of ``forge_objects`` once and records what produced it in
``sd_model.current_lora_hash`` -- a string built from the network names, their
text-encoder and UNet multipliers and any dynamic dimensions. The next call
compares that string and returns early when it matches, so an unchanged LoRA
selection costs nothing after the first application.

Because Model Chain caches whole ``sd_model`` objects rather than reloading
them, that hash travels with the cached model, and the early return keeps
working across a warm swap. So there is nothing to build here: the prepared
state is preserved by *not breaking* the host's own mechanism, and the job of
this module is to say when preserving it is safe and to force a rebuild when it
is not. That is the right shape for it. Reimplementing the loader to move
patched state around ourselves is exactly what the design rules out, and the
host is the only thing that knows which of its several LoRA paths a given
backend took.

Not every backend is safe to preserve. Nunchaku models fold LoRA weights into a
quantised kernel rather than into a patcher clone, and Forge rebuilds that state
rather than moving it -- so for those the honest answer is to invalidate and let
the host do its normal work.


Keeping a LoRA in its own stage
-------------------------------
The second problem is upstream of any of that. ``processed.all_prompts`` holds
the prompt as the user typed it, extra-network tags included -- the host strips
them from a *copy* on its way to the text encoder, which is why the tag still
shows up in infotext. Stage 2 inherits that prompt verbatim in Inherit mode and
as its first half in Append mode, and hands it to an ordinary img2img pass,
which parses the tags and applies them against Model B.

That is a Stage 1 LoRA silently becoming a Stage 2 LoRA, and it is at its worst
in the case the extension exists for: the two models are different
architectures, so the LoRA either fails to apply or applies to the wrong tensors.
``strip_networks`` removes the tags from the inherited half of the prompt only.
Anything typed into the Stage 2 boxes is untouched, which is how a Stage 2 LoRA
is meant to be requested.
"""

from __future__ import annotations

import logging
import re
import sys

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


# --------------------------------------------------------------------------- #
# Stage isolation
# --------------------------------------------------------------------------- #

RE_EXTRA_NET = re.compile(r"<(\w+):([^>]+)>")
"""Extra-network tag, spelled exactly as ``modules.extra_networks`` spells it.

Deliberately not a ``<lora:...>``-only pattern. ``<lyco:...>``, ``<hypernet:...>``
and anything else registered as an extra network are all Stage-1-model-specific
in the same way and for the same reason.
"""

_REPEATED_SEPARATORS = re.compile(r"\s*,(?:\s*,)+")
_RUNS_OF_SPACES = re.compile(r"[ \t]{2,}")


def strip_networks(text: str) -> tuple[str, list[str]]:
    """Remove extra-network tags from a prompt inherited from Stage 1.

    Returns the cleaned prompt and the tags that were dropped, so the caller can
    say what happened rather than silently changing the user's prompt.

    Only ever applied to the *inherited* half of a Stage 2 prompt. The Stage 2
    prompt boxes are left alone: a tag typed there is a deliberate request for a
    LoRA against Model B, and the host's parser is the only thing that should
    ever interpret it (section 5.4).
    """
    if not text:
        return text, []

    removed = [match.group(0) for match in RE_EXTRA_NET.finditer(text)]
    if not removed:
        return text, []

    return _tidy(RE_EXTRA_NET.sub("", text)), removed


def _tidy(text: str) -> str:
    """Close the gap a removed tag leaves behind.

    ``"a, <lora:x:1>, b"`` would otherwise become ``"a, , b"``, and a prompt
    that was nothing but a tag would become a lone comma -- which is not an
    empty prompt to the text encoder.
    """
    text = _REPEATED_SEPARATORS.sub(",", text)
    text = _RUNS_OF_SPACES.sub(" ", text)
    return text.strip().strip(",").strip()


# --------------------------------------------------------------------------- #
# Prepared state
# --------------------------------------------------------------------------- #

HASH_ATTRIBUTE = "current_lora_hash"
"""Where Forge records what produced a model's currently applied LoRA state."""

REBUILD = "<model-chain: rebuild>"
"""Value written to force the host to rebuild the state on its next use.

Any real hash is ``str([names, te_multipliers, unet_multipliers, dyn_dims])``,
so this cannot collide with one. Writing it is always the conservative
direction: at worst the host redoes work it could have skipped. Writing a
*stale* hash would be the dangerous direction, and nothing here ever does that.
"""

REBUILT_BY_HOST = ("nunchaku",)
"""``dynamic_args`` flags whose LoRA path Forge rebuilds rather than moves.

Nunchaku applies LoRA into a quantised kernel rather than into a patcher clone.
Its prepared state is not the movable object the rest of this module assumes,
so it is never claimed as preserved -- the host reapplies, as it would without
this extension.
"""


def state_of(sd_model) -> str | None:
    """The host's identifier for the LoRA set applied to ``sd_model``.

    None means "nothing to preserve": either the host has no LoRA loader, or
    this model has never had one applied. Both are ordinary.
    """
    try:
        value = getattr(sd_model, HASH_ATTRIBUTE, None)
    except Exception:
        return None

    if value in (None, "", REBUILD):
        return None
    return str(value)


def describe(state: str | None) -> str:
    """Short console description of a prepared LoRA state."""
    if not state:
        return "no LoRA applied"
    return "LoRA state ready"


def is_preservable(flags: dict | None) -> tuple[bool, str]:
    """Whether a model with these loader flags may keep its prepared state.

    ``flags`` is a ``MODEL_FLAGS`` snapshot -- the same ``dynamic_args`` fields
    the cache already carries with each model, which is exactly the record of
    which LoRA path the host took for it.
    """
    for flag in REBUILT_BY_HOST:
        if (flags or {}).get(flag):
            return False, f"{flag} rebuilds its LoRA state rather than moving it"
    return True, ""


def invalidate(sd_model, reason: str = "") -> bool:
    """Make the host rebuild this model's LoRA state before it is next used.

    Returns True when something was actually invalidated. False means there was
    no state to invalidate, which is not a failure -- a host without the LoRA
    extension, or a model that has never been patched, both land there.

    The module-level fallback exists because a host that tracked the hash
    globally rather than per-model would leave a *stale* global describing a
    model that is no longer loaded, and the whole point of this function is that
    no stale hash survives it. It only ever writes to an attribute that is
    already there, so it is inert on a host that has no such global.
    """
    invalidated = False

    if sd_model is not None and hasattr(sd_model, HASH_ATTRIBUTE):
        try:
            setattr(sd_model, HASH_ATTRIBUTE, REBUILD)
            invalidated = True
        except Exception:
            logger.debug("Model Chain: could not invalidate the prepared LoRA state", exc_info=True)

    host = sys.modules.get("networks")
    if host is not None and hasattr(host, HASH_ATTRIBUTE):
        try:
            setattr(host, HASH_ATTRIBUTE, REBUILD)
            invalidated = True
        except Exception:
            logger.debug("Model Chain: could not invalidate the host's LoRA hash", exc_info=True)

    if invalidated and reason:
        logger.info("Model Chain: the LoRA state will be rebuilt — %s", reason)

    return invalidated
