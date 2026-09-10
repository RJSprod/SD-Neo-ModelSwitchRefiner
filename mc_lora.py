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


The prompt Stage 2 is allowed to inherit
----------------------------------------
``strip_networks`` is a pattern, and a pattern can only remove what it can
recognise. Krea's literal commands (``[[...]]``) can carry *anything* a user
wants delivered to the Stage 1 image pipeline unchanged -- a reference
instruction about an ImageStitch image, another extension's macro, a wildcard --
and none of that is meaningful to a Stage 2 model that has a different text
encoder, different references and possibly a different architecture.

So the answer there is not a better pattern. Krea's pipeline builds *two*
finished prompts, one with its literals restored and one that never had them,
and leaves the second on the processing object with :func:`remember_inheritable`
for Stage 2 to read. Searching the first for the payloads afterwards is what
this deliberately avoids: a user who writes ``[[red hat]]`` over a scene the
Creative Writer independently described as having a red hat would lose the
writer's words along with their own.

``strip_networks`` stays, on top of it, as defence in depth -- a bare
``<lora:...>`` typed outside a literal command is still a Stage 1 tag, and is
still removed the way it always was.
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


INHERITABLE = "mc_stage1_inheritable_prompts"
"""Where a Stage 1 prompt-writing feature leaves what Stage 2 may inherit.

An attribute on the host's processing object, because that object is the one
thing both halves of this extension are handed for one generation and neither
of them owns. It is set in ``before_process`` by whatever wrote the prompt and
read in ``process`` by Model Chain; a generation nothing wrote a prompt for
never has it, and Stage 2 inherits ``all_prompts`` exactly as it always did.

A pair rather than one string: the negative prompt can carry literal commands
too, no language model ever sees it, and a Stage 1 negative that mentions
another extension's syntax has no more business in Stage 2 than a positive one.
"""


def remember_inheritable(p, positive: str = "", negative: str = "") -> None:
    """Leave the prompts Stage 2 may inherit on this generation.

    Never fatal, and never partial: a host object that will not take an
    attribute leaves Stage 2 doing what it did before, which is inheriting the
    finished prompt. That is the old behaviour rather than a broken one.
    """
    try:
        setattr(p, INHERITABLE, (str(positive or ""), str(negative or "")))
    except Exception:
        logger.debug("Model Chain: could not record the inheritable Stage 1 prompt",
                     exc_info=True)


def stage1_inheritable(p) -> tuple[str, str]:
    """``(positive, negative)`` Stage 2 may inherit, or two empty strings.

    Empty means "nobody rewrote this generation's prompt", which is the ordinary
    case and is not a failure: the caller falls back to ``all_prompts``, which is
    what it read before any of this existed.
    """
    try:
        found = getattr(p, INHERITABLE, None)
    except Exception:
        return "", ""
    if not isinstance(found, (tuple, list)) or len(found) != 2:
        return "", ""
    return str(found[0] or ""), str(found[1] or "")


# --------------------------------------------------------------------------- #
# Prepared state
# --------------------------------------------------------------------------- #

HASH_ATTRIBUTE = "current_lora_hash"
"""Where Forge records what produced a model's currently applied LoRA state."""

NO_NETWORKS = str([])
"""What a freshly loaded model carries before any LoRA has been applied.

``backend/diffusion_engine/base.py`` opens every engine with
``self.current_lora_hash = str([])``, and the LoRA loader compares against that
literal to decide whether it may return early. So "``[]``" is not a hash of
anything -- it is the host spelling "no networks", and reading it as a prepared
state was reporting *LoRA state ready* over a model with no LoRA in it at all.
"""

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

    if value in (None, "", REBUILD, NO_NETWORKS):
        return None
    return str(value)


def describe(state: str | None) -> str:
    """Short console description of a prepared LoRA state."""
    if not state:
        return "no LoRA applied"
    return "LoRA state ready"


def composite(*texts) -> str:
    """The extra-network composite these prompts ask the host to bake, as a key.

    Not a hash of anything Forge computes, and deliberately not an attempt to
    reproduce one -- ``current_lora_hash`` is the host's own record of what it
    *has* applied, and this is a statement about what the pass is about to ask
    for. Two of these compare equal when the pass will leave the baked weights
    alone, and differ when it will not.

    Why the answer matters more than it looks
    -----------------------------------------
    With ``on_the_fly`` false -- the default -- Forge does not hold a LoRA
    beside the weights, it merges it *into* them, and it records which merge
    each loaded model is carrying::

        backend/patcher/base.py     model.current_weight_patches_uuid = self.patches_uuid

    The LoRA loader applies its patches to a *clone* of the patcher, so a change
    to the composite mints a new ``patches_uuid``, and the next load compares
    the two::

        unpatch_weights = (self.model.current_weight_patches_uuid is not None
                           and self.model.current_weight_patches_uuid != self.patches_uuid)

    A mismatch is not a small correction. ``unpatch_model`` moves the whole
    model to the offload device and ``load`` brings all of it back with the new
    merge applied, so changing one LoRA weight from 1.0 to 1.3 costs a round
    trip of the entire UNet across the bus. From a user's log, the same
    checkpoint on the same card::

        [LORA] Loaded ... at weight 1.0     Moving model(s) has taken 106.95 seconds
        (no LoRA line -- unchanged prompt)  8/8 [00:06<00:00,  1.15it/s]
        [LORA] Loaded ... at weight 1.3     Moving model(s) has taken 10.73 seconds
        [LORA] Loaded ... four LoRAs        Moving model(s) has taken 70.54 seconds

    Every slow generation in that session is a line with a LoRA in it and every
    fast one is a line without.

    The first entry of that table is the one this function exists for. Nothing
    was baked into those weights yet, because a warm-up had just placed them --
    so the pass had to take all of them off the card and put them back, and the
    warm-up's twenty seconds bought a hundred seconds of undoing.

    Order is kept, and weights count
    --------------------------------
    ``names`` reaches the host's hash as a list in prompt order, so a reordered
    prompt rebuilds and this has to say so; sorting for tidiness would make two
    genuinely different requests compare equal. The multipliers are inside the
    tag and are part of it for the same reason: a weight change is a different
    merge, and the log above is what a weight change costs.

    Case and inner spacing are normalised because ``<LoRA:foo:1>`` and
    ``<lora:foo:1>`` are one request to Forge, and reading them as two would
    hold a warm-up back for a difference that does not exist.
    """
    found: list[str] = []
    for text in texts:
        for match in RE_EXTRA_NET.finditer(str(text or "")):
            found.append(_RUNS_OF_SPACES.sub(" ", match.group(0)).strip().casefold())
    return "\n".join(found)


def will_rebake(sd_model, *texts) -> str:
    """Why the pass about to run is *certain* to re-merge the weights, or "".

    Answers one question and refuses to guess at the rest, because the two
    mistakes are not symmetric. Saying "it will re-merge" when it will not costs
    a warm-up that had nothing to warm; saying "it will not" when it will costs
    the round trip this exists to avoid, on a card with no room for it.

    So only the two cases nothing has to be inferred for are reported:

    *nothing is applied and the prompt asks for something.* The host's
    ``current_lora_hash`` is ``str([])`` and ``compiled_lora_targets_hash`` will
    not be, so ``process_network_files`` cannot take its early return. Every
    tag in the prompt becomes a patch on a *clone* of the patcher, the clone's
    ``patches_uuid`` differs from the ``current_weight_patches_uuid`` stamped on
    the model by whatever loaded it, and ``partially_load`` responds to that by
    unpatching the model to the offload device and loading all of it back.

    *something is applied and the prompt asks for nothing.* The mirror image,
    and it costs exactly the same: the composite becomes ``str([])``, the hash
    still differs, and the weights still make the round trip. A user who deletes
    a LoRA tag pays what a user who adds one pays.

    Everything else returns "". Two non-empty composites may or may not be the
    same set at the same weights, and telling them apart means resolving names
    to filenames the way the host's own loader does -- which is the
    reimplementation this module exists not to do. The case is also the cheap
    one: a model that already carries a merge is a model that has been through a
    generation, so its weights are on the card and there is nothing for a
    warm-up to place.

    ``sd_model`` is the loaded model; ``texts`` are the prompts the pass will
    hand the host, positive and negative -- both, because the host's extra
    networks pass reads both.
    """
    applied = state_of(sd_model)
    wanted = composite(*texts)

    if applied is None and wanted:
        return ("the prompt asks for an extra network and nothing is applied yet, "
                "so the host will merge it into these weights")
    if applied is not None and not wanted:
        return ("an extra network is merged into these weights and the prompt no "
                "longer asks for one, so the host will unmerge it")
    return ""


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
