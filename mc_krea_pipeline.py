"""The Krea 2 prompt pipeline: one ordered path, two independent features.

Creative Mode writes a scene. Spatial Layout places things in one. They used to
be one feature with a switch inside it -- the Spatial controls were built into
Creative Mode's group, hidden when it was off, and the hook that ran both
returned immediately unless Creative Mode was on. That made Spatial a mode of
Creative rather than a peer of it, and the two questions "should a language
model write my prompt" and "where do the subjects go" have nothing to do with
each other.

This module is the answer to "if they are independent, what decides the order".

    source = the prompt as typed, minus its [[literal commands]]
    scene  = source, or what the Creative Writer made of it
    final  = scene, or the structured prompt the compositor built around it
    model  = final, with the literal commands restored around it

The literals are the fourth line
--------------------------------
Every prompt above is the *transformable* one. A ``[[literal command]]`` never
appears in any of them: it is lifted out before ``source`` exists and put back
once, into ``model``, after the last pass has finished. That is why this module
can be read without thinking about them -- and why the one place they are
restored is the same place ``final`` is decided, so no fallback path can insert
one twice or forget one.

Two finished prompts leave here, not one. ``model`` is what Stage 1 generates
from; ``final`` is what Stage 2 may inherit, and is the representation no
payload was ever written into. See :class:`mc_creative_krea.Prepared`.

Six combinations, one path
--------------------------
=====================  ==============================================
Neither                nothing happens; the typed prompt is generated
Creative only          source -> Writer -> final
Spatial Direct only    source -> compositor -> final
Spatial Smart only     source -> Composer -> compositor -> final
Creative + Direct      source -> Writer -> compositor -> final
Creative + Smart       source -> Writer -> Composer -> compositor -> final
=====================  ==============================================

Every one of those is the same function below with different branches taken.
That is deliberate: two features that are independent in the *UI* still have to
be ordered in the *pipeline*, and the reliable way to order two things is to
have one place that does it. Two Forge script hooks whose execution order
happens to be right is not an ordering, it is a coincidence with a test.

What this module is not
-----------------------
It is not a scheduler and it owns no host state. It does not know about the
progress bar, the broker, re-entrancy or ``p.extra_generation_params``; the hook
in ``scripts/model_chain_krea_creative.py`` owns all of those and calls this
between them. It does not know how to run the Creative Writer either -- that
needs an event loop, a progress bar and somewhere to put a complaint, all of
which belong to the hook -- so the writer arrives as a callable.

Feature modules stay independent underneath it: :mod:`mc_creative_krea` knows
nothing about layouts, :mod:`mc_spatial` knows nothing about rolls, and
:mod:`prompt_master.krea.spatial` knows nothing about either. This module is the
only place that knows both exist.

Failing forwards is not allowed
-------------------------------
Every failure here falls back to the thing that was true one step earlier, and
says so. A writer that will not answer leaves Spatial eligible and composing
around the raw prompt -- which is the change §21 asks for and the opposite of
what the old coupling did, where a failed roll took the boxes down with it. A
Composer that will not answer becomes a direct merge. A compositor that raises
leaves the best scene available. None of them refuses a generation, and none of
them silently drops a box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


# --------------------------------------------------------------------------- #
# What one generation asks for, and what it got
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Request:
    """Everything this pipeline needs to know about one press of Generate.

    Assembled by the hook out of the panel's controls and the parsed layout.
    Deliberately plain data: a test can build one without a page, a host or a
    checkpoint, which is how the six combinations above are checked.
    """

    source: str
    """The transformable prompt: what the user typed, minus its literals.

    What every pass in here reads, and the only prompt string any of them is
    given. Never rewritten in place.
    """

    raw_source: str = ""
    """Exactly what the user typed, ``[[...]]`` included, or "" when identical.

    Carried for the metadata and for nothing else -- no pass reads it, which is
    the property the whole feature rests on.
    """

    literals: object = None
    """The global :class:`prompt_master.krea.literals.LiteralParse`, or ``None``.

    A sidecar, not an input. It travels beside the pipeline rather than through
    it and is opened once, at the end, by the assembly step.
    """

    creative: bool = False
    """Whether the Creative Writer should run. Independent of ``layout``."""

    creative_settings: dict = field(default_factory=dict)
    """What the Creative panel is set to, for the metadata and the LoRA tail."""

    layout: object = None
    """A parsed :class:`prompt_master.krea.spatial.Layout`, possibly empty.

    Empty covers all three of "Spatial Layout is off", "nothing is drawn" and
    "this build cannot read that document", and all three mean the same thing
    here: there is nothing to compose, so the pipeline is whatever Creative Mode
    alone makes of it.
    """

    ratio: str = ""
    """The generation's aspect ratio, as a phrase the prompt names."""

    image_seed: int = 0
    """The host's seed for this image, or a negative number if it has not
    settled one yet. Only used as the Composer's seed basis when no Creative
    roll supplies one."""

    record_scenes: bool = True

    @property
    def regions(self) -> tuple:
        return tuple(getattr(self.layout, "regions", ()) or ())

    @property
    def spatial(self) -> bool:
        """Whether there is actually a composition to build."""
        return bool(self.regions)

    @property
    def commands(self) -> tuple:
        """The global literal commands, or nothing at all."""
        return tuple(getattr(self.literals, "commands", ()) or ())

    @property
    def wanted(self) -> bool:
        """Whether this generation is one this pipeline has any business in.

        Three reasons now rather than two. A prompt carrying nothing but
        ``[[<lora:krea2_edit:1>]] a portrait`` with both features switched off
        still has to have its brackets taken off before Forge sees it --
        otherwise the syntax means one thing with Creative Mode on and another
        with it off, which is the sort of inconsistency a user discovers at the
        worst possible moment.
        """
        return bool(self.creative or self.spatial or self.commands)


@dataclass
class Outcome:
    """What the pipeline did, and everything the hook has to say about it."""

    prepared: object = None
    """The finished prompt and metadata, or ``None`` to leave ``p.prompt`` alone.

    ``None`` is not a failure code. It is what "generate exactly what the user
    typed" looks like, and it is the right answer for a generation with neither
    feature on, for a writer that failed with no layout behind it, and for a
    compositor that raised with no written scene to fall back to.
    """

    roll: object = None
    ran_creative: bool = False
    ran_composer: bool = False
    ran_spatial: bool = False
    """Whether the compositor actually built the structured prompt.

    Distinct from "there was a layout": a compositor that raised leaves this
    false with a note beside it, and the hook says so on the result rather than
    claiming the boxes were applied.
    """

    creative_note: str = ""
    """Why the Creative Writer did not write this prompt, if it did not."""

    spatial_note: str = ""
    """Why the layout was not applied as asked, if it was not."""

    @property
    def merge(self) -> str:
        return "smart" if self.ran_composer else "direct"


# --------------------------------------------------------------------------- #
# The checkpoint guard, which belongs to the pipeline and not to one feature
# --------------------------------------------------------------------------- #


def objection() -> str:
    """Why this checkpoint must not be given a Krea 2 structured prompt, or "".

    Named here rather than left inside Creative Mode because it was never a
    Creative Mode rule: it is about what the *image model* can read. A Spatial
    Direct generation makes no language-model request at all and still hands
    Krea 2's structured JSON prompt to whatever checkpoint is loaded, so a
    Spatial-only generation needs exactly the same guard for exactly the same
    reason -- a paragraph of JSON handed to a model with 77 tokens of room does
    not look like a smaller version of the feature, it looks like a bug.

    The implementation still lives in :mod:`mc_creative_krea` because that is
    where it was written and moving it would be churn; what changes here is who
    is understood to own the rule.
    """
    import mc_creative_krea

    return mc_creative_krea.checkpoint_objection()


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


def run(request: Request, write=None) -> Outcome:
    """Source, scene, final prompt -- in that order, whichever features are on.

    ``write`` runs the Creative Writer and returns ``(roll, complaint)``: the
    roll when one was produced, and a sentence when one was not. It is a
    callable rather than an import because running the writer needs an event
    loop, a progress bar and a place to put the complaint, all of which are the
    hook's and none of which are this module's.

    The invariant, stated once: **Creative, when enabled, always runs before
    Spatial.** Everything else in here is a fallback.
    """
    from prompt_master.krea import literals as literals_module
    from prompt_master.krea import spatial as spatial_module

    import mc_creative_krea
    import mc_spatial

    out = Outcome()
    if not request.wanted:
        return out

    source = str(request.source or "")
    scene = source

    # ---- Creative, first when it runs at all ------------------------------ #
    if request.creative and write is not None:
        roll, complaint = write(source)
        out.creative_note = str(complaint or "")
        if roll is not None and str(getattr(roll, "expanded", "") or "").strip():
            out.roll = roll
            out.ran_creative = True
            scene = roll.expanded
        # And if it did not: the scene stays the source, and Spatial carries on
        # from it. This is §21, and it is the whole difference between "Spatial
        # is a mode of Creative" and "Spatial is a feature". The user asked for
        # boxes; a copy-editor being unavailable is not a reason to refuse them.

    settings = dict(request.creative_settings) if out.ran_creative else {}

    if not request.spatial:
        if not out.ran_creative and not request.commands:
            # Nothing was produced and nothing should be substituted: the prompt
            # the user typed is already in the box.
            return out
        # With no writer behind it this is the typed prompt with its brackets
        # taken off and its payloads moved to the ends -- which is the whole of
        # what the feature promises when neither language-model pass ran, and is
        # also the right answer when the writer failed with literals in hand.
        out.prepared = _prepared(request, out, body=scene, settings=settings)
        return out

    # ---- Spatial, on whatever scene it was handed ------------------------- #
    #
    # The one line §7 is about: Spatial takes a *scene string*, not "a Creative
    # result". Where that string came from is not its business, which is why
    # this works identically with the writer on and with it off.
    input_scene = scene
    background = ""
    composed = None

    if request.layout.compose_mode == spatial_module.SMART:
        composed = mc_spatial.compose(
            source=source, scene=input_scene, layout=request.layout,
            ratio=request.ratio,
            seed=mc_spatial.composer_seed_for(
                creative_seed=out.roll.creative_seed if out.ran_creative else None,
                image_seed=request.image_seed),
            reserve=mc_creative_krea.image_reserve_bytes())

        # The last language-model phase of this plan has now finished, so this
        # is where the card goes back to the image side. §13: which phase is
        # last depends on the configuration, and only two of the six put the
        # Composer there -- but in both of them the roll deliberately did not
        # hand back, because between the writer and the Composer the only way to
        # free image VRAM is to stop the server the Composer is about to use.
        #
        # Run whether or not the Composer succeeded. A Composer that failed
        # still leaves the same card behind it, and the image pass that follows
        # needs the same room either way.
        freed = mc_creative_krea.hand_back_vram()
        if freed:
            logger.info("Model Chain: freed %.1f GB for the image generation that "
                        "follows the Spatial Composer", freed / (1024 ** 3))

        if composed.ran:
            scene, background = composed.scene, composed.background
            out.ran_composer = True
            logger.info("Model Chain: Spatial Composer reconciled the %s scene",
                        "enhanced" if out.ran_creative else "raw")
        else:
            # Direct merge is the answer to every Composer failure. The boxes
            # are the user's and are unaffected; what is lost is the
            # de-conflicting of the global scene, and saying so is the
            # difference between a fallback and a mystery.
            logger.warning("Model Chain: the Spatial Composer did not run (%s); "
                           "merging the layout directly instead", composed.failed)
            out.spatial_note = ("the scene was merged directly because the Spatial "
                                f"Composer did not run — {composed.failed}")
            scene = input_scene

    try:
        # Global literals go inside ``high_level_description``, so the document
        # the image model reads stays one valid structured prompt rather than a
        # JSON object with tags loose around it. Region literals are already in
        # their own element's desc, put there by the compositor.
        prompt = spatial_module.compose(
            request.layout, scene=literals_module.restore(scene, request.literals),
            background=background, ratio=request.ratio)
        # The same document, built again from the same validated layout with
        # every payload left out. Built rather than edited: see
        # :attr:`mc_creative_krea.Prepared.inheritable`.
        inheritable = spatial_module.compose(request.layout, scene=scene,
                                             background=background,
                                             ratio=request.ratio, literals=False)
    except Exception as exc:
        from modules import errors

        errors.report("Model Chain: the spatial compositor failed", exc_info=True)
        out.spatial_note = ("the Spatial Layout was not applied because the "
                            f"compositor failed ({exc})")
        # The best scene still available. With a roll behind it that is the
        # writer's paragraph, which is worth substituting; without one it is the
        # prompt the user typed, which is already in the box -- unless it had
        # literal commands in it, which have to come off it either way.
        #
        # The region literals are lost on this path and cannot be otherwise:
        # they belong to elements of a document that could not be built. Saying
        # so above is the honest half; inventing somewhere else to put them
        # would move a BBOX-local command into the global scene, which is the
        # one thing §21 forbids outright.
        if out.ran_creative or request.commands:
            out.prepared = _prepared(request, out, body=scene, settings=settings)
        return out

    metadata = mc_spatial.metadata(
        request.layout, compose_mode=request.layout.compose_mode, composed=composed,
        input_scene=input_scene, record_scenes=request.record_scenes,
        creative=out.ran_creative)
    logger.info("Model Chain: Spatial Layout applied — %s scene, %s element%s, "
                "%s merge, %s characters of structured prompt",
                "enhanced" if out.ran_creative else "raw",
                len(request.regions), "" if len(request.regions) == 1 else "s",
                request.layout.compose_mode, f"{len(prompt):,}")
    out.ran_spatial = True
    out.prepared = mc_creative_krea.prepare(
        out.roll, settings, prompt=prompt, spatial=metadata,
        inheritable=inheritable, literals=request.literals)
    return out


def _prepared(request: Request, out: Outcome, body: str, settings: dict):
    """One finished pair of prompts for a generation with no structured prompt.

    The two paths that reach it -- no layout, and a compositor that raised --
    want the same two strings: the body with the global literals restored around
    it for Stage 1, and the body untouched for Stage 2. Written once so that
    "restored exactly once" is a property of there being one assembly step
    rather than of two of them agreeing.
    """
    from prompt_master.krea import literals as literals_module

    import mc_creative_krea

    return mc_creative_krea.prepare(
        out.roll, settings,
        prompt=literals_module.restore(body, request.literals),
        inheritable=body, literals=request.literals)


# --------------------------------------------------------------------------- #
# What the panel says the pipeline will do
# --------------------------------------------------------------------------- #


def described(*, creative: bool, spatial: bool, mode: str) -> str:
    """One sentence naming the pipeline the two toggles actually describe.

    §4.4. The four Spatial sentences differ only in where the scene comes from
    and whether a second request is made, and both of those are decided by a
    checkbox somebody else's section owns -- which is exactly why the sentence
    has to be computed from both rather than written under one.
    """
    from prompt_master.krea import spatial as spatial_module

    if not spatial:
        return ("Spatial Layout is off. "
                + ("Creative Mode writes the prompt and nothing is composed onto it."
                   if creative else
                   "The prompt is generated exactly as typed."))

    smart = str(mode or "").strip().casefold() == spatial_module.SMART
    if smart and creative:
        return ("Smart Spatial Compose: Creative Mode writes the scene first, then "
                "the Spatial Composer reconciles it with your regions before the "
                "BBOX prompt is built. Two language-model requests.")
    if smart:
        return ("Smart Spatial Compose: your prompt goes straight to the Spatial "
                "Composer, which reconciles the scene with your regions before the "
                "BBOX prompt is built. One language-model request; Creative Mode is "
                "not involved.")
    if creative:
        return ("Direct BBOX Merge: Creative Mode writes the scene first, then your "
                "regions are applied deterministically without a second request.")
    return ("Direct BBOX Merge: your prompt is used exactly as typed as the global "
            "scene and your regions are applied deterministically. No "
            "language-model request is made — the fastest and most predictable "
            "Spatial option.")
