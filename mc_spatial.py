"""Krea Creative Mode Spatial BBOX: the host side of the layout.

Creative Mode turns a short idea into a rich scene. Spatial Layout adds the
other half of a composition -- where things go -- and it adds it *beside* the
writer rather than through it:

    source prompt
        -> Creative Director            (local, no model)
        -> LLM pass 1: Creative Writer  -> enhanced scene
        -> LLM pass 2: Spatial Composer -> reconciled scene   (optional)
        -> deterministic BBOX compositor                      (code)
        -> pinned LoRA tags
        -> p.prompt

The regions never enter pass 1. They are the user's own words about their own
picture, and a rewriter that "improved" them would leave the compositor placing
the original wording beside an improved copy of it -- which is how one subject
becomes two.

What lives where
----------------
:mod:`prompt_master.krea.spatial` is the pure half: parsing, validation, the
deterministic hints and the compositor. It imports nothing from this extension
and can be reasoned about on its own, which is what the acceptance tests need.
:mod:`prompt_master.krea.composer` is pass 2's instruction and the strict reader
for what comes back. This module is the host glue -- the preferences, the run
that drives pass 2 through :mod:`mc_llm_sessions`, and what one spatial
generation records about itself.

Off is off
----------
Every path below is reached only when the user turned Spatial Layout on *and* a
region survived validation. With the feature off, Creative Mode assembles the
same user turn it always did, makes the same one request, and produces the same
bytes -- the placement note is not added, no second request is made, no
infotext key is written, and llama.cpp's prompt cache sees the same prefix it
saw yesterday. That is a property this module is arranged to make testable
rather than a claim about it.
"""

from __future__ import annotations

import logging

import mc_llm_progress
import mc_llm_sessions as sessions
import mc_llm_state
import mc_progress

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ENABLED = "krea_spatial_enabled"
COMPOSE_MODE = "krea_spatial_compose_mode"
LAYOUT = "krea_spatial_layout"
RECORD_SCENES = "krea_spatial_record_scenes"
KLEIN_MODE = "klein_spatial_mode"
BACKEND = "spatial_backend"
REGION_STEPS = "klein_spatial_region_steps"
"""The preferences this feature owns. All seven are its own keys.

The layout is persisted, and that is deliberate rather than incidental: boxes
are minutes of work with a mouse, and a WebUI restart that quietly emptied the
canvas would be exactly the silent loss of layout state the design intent
forbids. It is *not* part of a Creative profile -- a profile describes how art
direction behaves, and a composition is about one picture.

``KLEIN_MODE`` is the odd one out and is named for its backend rather than for
Krea, because it is the only preference here that means nothing to Krea 2. There
is one canvas and one enabled switch; what differs between the two backends is
what they *do* with the canvas, so the compose mode is Krea's question and the
spatial mode is Klein's, and both are remembered so that switching checkpoints
back and forth does not lose either answer.
"""


def settings() -> dict:
    """Every Spatial preference, with defaults filled in.

    ``compose_mode`` defaults to Smart, which is what the design intent asks
    for and costs nothing until there is a region to compose around: Smart with
    an empty canvas makes no request, because there is nothing for pass 2 to
    reconcile the scene with.
    """
    from prompt_master import spatial as generic
    from prompt_master.krea import spatial

    try:
        stored = mc_llm_state.preferences()
    except Exception:
        logger.debug("Model Chain: could not read the Spatial Layout preferences",
                     exc_info=True)
        stored = {}

    mode = str(stored.get(COMPOSE_MODE) or spatial.SMART).strip().casefold()
    return {
        "enabled": bool(stored.get(ENABLED, False)),
        "compose_mode": mode if mode in spatial.COMPOSE_MODES else spatial.SMART,
        "layout": str(stored.get(LAYOUT) or ""),
        "record_scenes": bool(stored.get(RECORD_SCENES, True)),
        # Normalised on the way out rather than trusted: a preference file
        # written by a build that offered a mode this one does not must land on
        # Auto rather than reach the resolver as a string nothing matches.
        "klein_mode": generic.normalise_mode(stored.get(KLEIN_MODE), generic.AUTO),
        # Which backend the panel shows. Auto follows the checkpoint; the other
        # two are the user telling the page what they have, for the case where
        # a host's model chooser is not something this extension can read.
        "backend": _backend(stored.get(BACKEND)),
        # How much of the sample the Klein regions apply for. Klein's backend
        # costs one model evaluation per region per step, and composition is
        # settled long before the last step -- so this is the dial between the
        # two, and it is a real one rather than a placebo.
        "region_steps": _percent(stored.get(REGION_STEPS), 60),
    }


def _percent(value, fallback: int) -> int:
    try:
        return max(10, min(100, int(value)))
    except (TypeError, ValueError):
        return fallback


def _backend(value) -> str:
    """The stored backend preference, normalised, defaulting to Auto.

    Imported lazily so that :func:`settings` stays answerable on an installation
    where the Klein module could not load at all -- the Krea half of this feature
    has never needed it and must not start needing it now.
    """
    try:
        import mc_spatial_klein

        return mc_spatial_klein.normalise_backend(value)
    except Exception:
        logger.debug("Model Chain: could not read the Spatial backend preference",
                     exc_info=True)
        return "auto"


def remember(**values) -> None:
    """Keep a Spatial preference. Never fatal: this is a convenience."""
    try:
        mc_llm_state.remember(**values)
    except Exception:
        logger.debug("Model Chain: could not save the Spatial Layout preferences",
                     exc_info=True)


def layout_for(serialized, width=0, height=0, compose_mode="") -> object:
    """The layout this generation will use, with the panel's mode applied.

    The serialized state carries a ``compose_mode`` of its own because the
    editor writes the whole document, and the radio button on the panel is what
    the user is actually looking at -- the same precedence the Creative panel
    settles in :func:`scripts.model_chain_krea_creative._stored`, for the same
    reason. When the panel sent nothing, the document answers.
    """
    from prompt_master.krea import spatial

    parsed = spatial.parse(serialized, width=width, height=height)
    mode = str(compose_mode or "").strip().casefold()
    if mode in spatial.COMPOSE_MODES and mode != parsed.compose_mode:
        from dataclasses import replace

        return replace(parsed, compose_mode=mode)
    return parsed


# --------------------------------------------------------------------------- #
# Pass 2
# --------------------------------------------------------------------------- #


def composer_seed(creative_seed) -> int:
    """The Composer's seed, derived from the Creative seed rather than drawn.

    One recorded number reproduces the whole roll. A separately drawn seed would
    mean an image whose Creative seed is recorded and whose second pass is not
    reproducible -- and the second pass is exactly the one somebody comparing
    Smart against Direct needs to be able to run twice.
    """
    from prompt_master.krea import composer, director

    try:
        return director.stable_hash(int(creative_seed), composer.SEED_PURPOSE)
    except (TypeError, ValueError):
        return director.stable_hash(0, composer.SEED_PURPOSE)


NO_CREATIVE_SEED = 0
"""The Composer's seed basis when Creative Mode did not run.

Smart Spatial used to be reachable only through a Creative roll, so deriving its
seed from the Creative seed was both natural and complete: one recorded number
reproduced the whole generation. Spatial is a peer feature now and can run with
no roll behind it at all, so the derivation needs a basis that exists in that
case too.

The image seed is used when the host has settled one, and this constant when it
has not -- ``before_process`` runs before Forge resolves ``-1``, so a random
seed genuinely has no number yet. A fixed basis there is not a weakness: the
Composer reconciles a scene with a layout, which is a correction rather than a
creative draw, and the same scene over the same boxes *should* reconcile the
same way twice. What matters is that it is deterministic, independent of
Creative, and recorded whenever the pass actually ran -- which is all §11 asks
for.
"""


def composer_seed_for(creative_seed=None, image_seed=0) -> int:
    """The Composer's seed, whether or not Creative Mode ran before it.

    With a Creative roll behind it, exactly as before: derived from the Creative
    seed, so one recorded number still reproduces both passes. Without one,
    derived from the image seed when the host has settled one, and from
    :data:`NO_CREATIVE_SEED` when it has not.
    """
    if creative_seed is not None:
        return composer_seed(creative_seed)
    try:
        basis = int(image_seed)
    except (TypeError, ValueError):
        basis = NO_CREATIVE_SEED
    if basis < 0:
        basis = NO_CREATIVE_SEED
    return composer_seed(basis)


class Composed:
    """What pass 2 produced, or why it did not, in the form the hook reads."""

    def __init__(self, scene: str, background: str = "", failed: str = "",
                 seed: int = 0):
        self.scene = scene
        self.background = background
        self.failed = failed
        self.seed = int(seed)

    @property
    def ran(self) -> bool:
        return not self.failed


def compose(source: str, scene: str, layout, ratio: str, seed: int,
            reserve: int = 0, task_id: str = "") -> Composed:
    """Run the Spatial Composer once, and never raise.

    Returns the reconciled scene, or the scene it was given with a reason
    attached. **Direct BBOX Merge is the answer to every failure here** -- a
    timeout, an interrupt, an empty reply, a reply that is not the schema, a
    model that will not start. §5.5 is unusually blunt about it and it is right
    to be: the user asked for a picture, drew the boxes for it, and a copy-editor
    being unavailable is not a reason to refuse them one.

    The interrupt case is the one worth naming separately. It does not cancel the
    generation, because by the time this runs the generation is already running
    and the roll before it has already succeeded; pressing Stop during the
    composer pass skips the composer pass, and the host's own loop reads the same
    flag a moment later and stops the sampling if that is what was meant.
    """
    cancel = sessions.Cancellation()
    run = sessions.krea_compose(source, scene, layout, ratio, seed, cancel, reserve)
    progress = mc_llm_progress.reporter
    progress.begin(task_id, _compose_size(source, scene, layout, ratio), warm=None,
                   claim=False, kind=mc_llm_progress.COMPOSER)
    result = Composed(scene=scene, seed=seed)
    written = ""
    finished = False
    try:
        for event in run:
            if event.kind == sessions.DONE:
                data = event.data or {}
                result = Composed(scene=str(data.get("scene") or scene),
                                  background=str(data.get("background") or ""),
                                  seed=seed)
                written = result.scene
                finished = True
                break
            if event.kind in (sessions.FAILED, sessions.CANCELLED):
                result = Composed(scene=scene, seed=seed,
                                  failed=event.text or "the pass was stopped")
                break
            if event.kind == sessions.CHUNK:
                progress.enter(mc_progress.PHASE_KREA_WRITE)
                progress.wrote(event.text)
            elif event.kind == sessions.STATUS:
                progress.enter(_phase_for(event.text))
            if progress.interrupted():
                cancel.cancel()
                result = Composed(scene=scene, seed=seed,
                                  failed="the composition pass was stopped")
                break
    except Exception as exc:
        logger.debug("Model Chain: the Spatial Composer failed", exc_info=True)
        result = Composed(scene=scene, seed=seed,
                          failed=str(exc) or exc.__class__.__name__)
    finally:
        run.close()
        if finished:
            progress.end(written)
        else:
            progress.abandon()
    return result


def _compose_size(source: str, scene: str, layout, ratio: str) -> int:
    """How much text pass 2 is about to evaluate, in characters.

    Its whole request, instruction included, and unlike pass 1 that is not an
    approximation to be apologised for: the Composer's system message is not the
    one llama.cpp has cached, so on every run it really is read. See
    :mod:`prompt_master.krea.composer` for why it is a different message and what
    that costs.
    """
    from prompt_master.krea import composer

    return len(composer.SYSTEM_PROMPT) + len(
        composer.user_content(source, scene, layout, ratio))


def _phase_for(status: str) -> str:
    """Which bar phase a status line means, for the composer pass."""
    text = str(status or "").casefold()
    if "starting" in text or "waiting" in text or "preparing" in text:
        return mc_progress.PHASE_KREA_WAIT
    return mc_progress.PHASE_KREA_READ


# --------------------------------------------------------------------------- #
# What a spatial generation records about itself
# --------------------------------------------------------------------------- #


def metadata(layout, *, compose_mode: str, composed: Composed | None,
             input_scene: str = "", record_scenes: bool = True,
             creative: bool = True) -> dict:
    """The Spatial keys for one generation's infotext.

    Namespaced separately from the Creative keys and answering a different
    question. The image's own ``Prompt:`` line is still the whole answer to *how
    do I make this again* -- it is the finished structured prompt, assigned
    before Forge wrote the infotext -- and these are the answer to *how do I get
    back to the canvas I drew it on*.

    ``Krea Spatial Input Scene`` is recorded only in Smart mode, and that is the
    same reasoning §5.2 applied to the expanded prompt rather than a different
    one. In Direct mode the input scene *is* the ``high_level_description``
    inside the Prompt line, and a second copy would be a few hundred bytes
    repeating what the file already says. In Smart mode it is genuinely nowhere
    else: the Prompt carries the reconciled scene, and the text it was
    reconciled from would otherwise be unrecoverable -- which would make the A/B
    comparison the design intent asks for impossible to do after the fact.

    It used to be called ``Krea Enhanced Scene``, and that name stopped being
    true the day Spatial could run without Creative Mode: with the writer off,
    the scene handed to the Composer is the user's own sentence and nothing
    enhanced it. The neutral name says what the field *is* -- the scene supplied
    to the Composer before reconciliation -- in both cases. Old images keep the
    old key and are still read; nothing about exact replay depended on either,
    because the Prompt line is authoritative.

    ``creative`` says whether the Creative Writer ran. A Spatial-only image
    records no Creative keys at all, which is what makes its paste able to
    switch Spatial off without a Creative key being present to switch.
    """
    import mc_infotext
    from prompt_master.krea import composer, spatial

    recorded = {
        mc_infotext.SPATIAL_MODE: "True",
        mc_infotext.SPATIAL_VERSION: spatial.VERSION,
        mc_infotext.SPATIAL_PROMPT_VERSION: spatial.PROMPT_VERSION,
        mc_infotext.SPATIAL_COMPOSE_MODE: compose_mode,
        mc_infotext.SPATIAL_LAYOUT: layout.serialize(),
    }
    if not creative:
        recorded[mc_infotext.SPATIAL_SOURCE] = "prompt"
    if compose_mode == spatial.SMART and composed is not None and composed.ran:
        recorded[mc_infotext.SPATIAL_COMPOSER_SEED] = composed.seed
        recorded[mc_infotext.SPATIAL_COMPOSER_VERSION] = composer.INSTRUCTION_VERSION
        model = _composer_model()
        if model:
            recorded[mc_infotext.SPATIAL_COMPOSER_MODEL] = model
        if record_scenes:
            if input_scene:
                recorded[mc_infotext.SPATIAL_INPUT_SCENE] = input_scene
            recorded[mc_infotext.SPATIAL_SCENE] = composed.scene
    return recorded


def _composer_model() -> str:
    """Which language model reconciled the scene.

    The same question :func:`mc_creative_krea.writer_identity` answers about
    pass 1, and the same answer today -- one llama-server, one backbone, both
    passes. It is recorded separately anyway, because "both passes ran on the
    same model" is a fact about this version rather than a promise, and an image
    that recorded one identity for two passes would be wrong the day that
    changes.
    """
    import mc_creative_krea

    return mc_creative_krea.writer_identity()
