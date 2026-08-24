"""Krea Creative Mode: one explicit press, one creative roll, one image.

Creative Mode sits in front of Forge's Generate button and in front of LLM
Studio's *Generate Krea Prompt*, and it does the same thing in both places:

    source prompt
        -> lift the [[literal commands]] out of it            (no model)
        -> resolve the Creative seed
        -> local Creative Director picks the art direction    (no model)
        -> exactly ONE Krea writer request
        -> one expanded Krea prompt
        -> the literal commands restored around it            (txt2img only)
        -> native Forge generation                            (txt2img only)

Nothing here starts on its own. There is no idle timer, no observer on the
prompt box, no repeat loop and no state machine counting down to anything. A
roll happens because somebody pressed a button, and the button they pressed is
the one they already knew about.

What replaced what
------------------
This module is the successor to the Krea Live controller, and it is smaller by
most of a file. Live's cache, revision counter, cooperative cancellation of
in-flight writes, reroll scheduler and failure circuit breaker all existed to
manage work that started without being asked for. When every roll is explicit,
all of that is answerable by "the user pressed it again": there is nothing to
debounce, nothing to invalidate, and no loop that could run away.

The roll happens inside the generation
--------------------------------------
It did not always. Creative Mode was first built so that the roll ran in a
Gradio handler *before* the native Generate click, because
``mc_llm_sessions`` takes the broker's workload lock for a whole run and waits
while the host is generating -- so an expansion asked for from inside a Forge
processing hook was an LLM run waiting for the image job that was waiting for
it. The ordering was arranged in the browser: intercept the click, roll, then
click Generate again.

That made every Creative generation depend on a live page. A press started a
roll and nothing else; the thing that actually started the image was a
``click()`` from JavaScript, minutes later, from a timer that browsers throttle
to once a second in a hidden tab and once a minute in a frozen one. Switch
windows and the image was late. Close the tab and it never came at all, which
is not a browser being unhelpful -- it is the design saying the job cannot
finish without one.

So the deadlock was addressed where it was, rather than routed around. A roll
that the image job is blocked waiting for is not competing with that job, and
``mc_broker.host_job()`` is how a caller says exactly that, for exactly as long
as it is true. The roll now runs in ``before_process``, one press starts
everything, and the browser is a spectator: close it after pressing Generate and
Forge finishes the job and writes the files, the same as any other generation.

The arming token did not survive it, and did not need to. It was permission for
one generation to spend a roll made before that generation started; with the
roll made *by* the generation, the hook that writes the prompt is the hook that
applies it, on one thread, in one call. What used to be a token is now the shape
of the code.

Four prompts, and only one of them is on screen
-----------------------------------------------
``source`` is what the user typed and keeps editing; it stays in the txt2img box
untouched. The writer never sees it -- it sees ``transform_source``, which is
``source`` with every ``[[literal command]]`` lifted out of it. ``expanded`` is
the writer's one result. ``generation`` is that with the literal commands
restored around it, and it exists only between :func:`prepare` and the line in
the processing hook that assigns it.

There is a fifth that is never on screen and never in a file: ``inheritable``,
the same finished prompt built without the literals, which is the only thing
Stage 2 is allowed to inherit. See :class:`Prepared`.

What replaced the pinned LoRAs
------------------------------
Creative Controls used to carry a Pinned LoRAs box whose contents were parsed
for ``<...>`` tags and appended to every generation. It is gone, and the literal
syntax is what replaced it: ``[[<lora:krea2_edit:1>]]`` in the prompt does the
same thing, in the place the tag belongs, without a second prompt input that
only accepted one kind of syntax. Nothing migrates it -- an image that recorded
``Krea Pinned LoRAs`` still shows the tags it used under "what the pasted image
records", and they are not applied to anything.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import mc_llm_progress
import mc_llm_sessions as sessions
import mc_llm_state
import mc_progress

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

IDLE = "Idle"
DIRECTING = "Choosing the art direction"
WRITING = "Writing the Krea prompt"
READY = "Prompt ready"
ERROR = "Error"


# --------------------------------------------------------------------------- #
# One roll's result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Roll:
    """Everything one creative roll produced, and what it took to produce it."""

    recipe: object
    source: str
    """The text the writer was given: the user's prompt minus its literals.

    Named ``source`` still, because that is what it is from the writer's point
    of view and because it is what the Creative metadata has always recorded.
    :attr:`raw_source` is the other half of the answer when the two differ.
    """

    expanded: str
    raw_source: str = ""
    """Exactly what the user typed, ``[[...]]`` included, or "" when identical.

    Recorded rather than reconstructed. The literal payloads and the clean text
    could in principle be reassembled into something close to the original, but
    only *close*: the user's own ordering, spacing and line breaks are theirs,
    and "Restore Creative setup" is supposed to hand back the prompt they wrote.
    """

    @property
    def typed(self) -> str:
        """What the user actually typed, for metadata and for restoring."""
        return self.raw_source or self.source

    @property
    def creativity(self) -> int:
        return int(getattr(self.recipe, "creativity", 0))

    @property
    def creative_seed(self) -> int:
        return int(getattr(self.recipe, "creative_seed", 0))

    @property
    def llm_seed(self) -> int:
        return int(getattr(self.recipe, "llm_seed", 0))


@dataclass(frozen=True)
class Prepared:
    """One roll, in the form the processing hook substitutes.

    Everything that hook needs and a :class:`Roll` does not carry: the prompt
    with the pinned networks on the end, and the values to write into infotext so
    the image can be explained afterwards.

    This used to be called ``Armed`` and used to carry a token, because the roll
    was made somewhere else -- a Gradio handler, ahead of the click -- and the
    token was that handler's permission for exactly one generation to spend the
    result. The roll happens inside the generation now, so there is no handover
    to authorise: the hook that makes this is the hook that applies it, on the
    same thread, in the same call.
    """

    roll: Roll | None = None
    """The Creative roll this prompt was written from, or ``None``.

    ``None`` is a Spatial-only generation -- the boxes were composed around the
    prompt exactly as typed and no writer ran -- or a generation whose only
    change was that its literal commands were put back. It is not a degraded
    Creative generation and must not record itself as one: an image carrying
    ``Krea Creative Mode`` would tell a later paste to switch off a feature that
    was never on, and would tell a reader that a language model wrote a sentence
    the user typed.
    """

    generation: str = ""
    """The prompt Stage 1 generates from: the finished text, literals restored."""

    inheritable: str = ""
    """The same prompt with no literal payload ever written into it.

    The one thing Stage 2 may inherit, and the reason it is built rather than
    derived. Removing the payloads from :attr:`generation` afterwards would mean
    searching the finished prompt for the strings that were put into it -- and
    a user who writes ``[[red hat]]`` over a scene the writer independently
    described as having a red hat would lose the writer's words too. Two
    representations, assembled separately, cannot make that mistake.

    Empty means "nothing to isolate": Stage 2 inherits the ordinary prompt, as
    it did before this existed.
    """

    literals: object = None
    """The parsed global sidecar for this generation, or ``None``.

    Held for the metadata count and for nothing else. Nothing downstream reads a
    payload out of here -- by the time a :class:`Prepared` exists the payloads
    are already in :attr:`generation`, exactly once.
    """

    settings: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)
    """The Spatial keys for this generation, or nothing at all.

    Merged into :attr:`metadata` rather than written by a second hook, so that
    one generation writes its infotext in one place and a spatial image cannot
    end up carrying half a record.
    """
    """The configuration this roll ran with, for the metadata to record.

    Read off the panel at press time rather than out of the preferences file,
    because they can differ: the file holds what was last saved and the panel
    holds what the user is looking at. What made the picture is the second one.
    """

    @property
    def metadata(self) -> dict:
        """What this generation records about how its prompt was written.

        Two questions, and the fields answer them separately. *How do I make this
        picture again* is answered by the image's own ``Prompt:`` line, which is
        the expanded prompt Creative Mode substituted -- so an ordinary paste
        reproduces the image by restoring that prompt and turning Creative Mode
        off, and needs nothing from here at all. *How do I get back to the
        workflow that produced it* is what these fields are for: the source
        phrase, the position, both seeds, the recipe as compact ids, the axis
        configuration that allowed it, and the writer that wrote it.

        The expanded prompt is still absent, and for the reason it always was: it
        *is* the ``Prompt:`` line. Writing the paragraph a second time under a key
        of ours would add a few hundred bytes to every PNG to repeat what the file
        already says, and create a second copy that a later paste could disagree
        with.
        """
        import mc_infotext

        # A Spatial-only generation records the Spatial half and nothing else.
        # There is no roll to describe, and describing one anyway is how a paste
        # ends up switching off a feature that never ran.
        if self.roll is None:
            found = dict(self.spatial or {})
            found.update(self.literal_metadata)
            return found

        recipe = self.roll.recipe
        recorded = {
            mc_infotext.CREATIVE_MODE: "True",
            mc_infotext.CREATIVE_CREATIVITY: self.roll.creativity,
            mc_infotext.CREATIVE_SEED: self.roll.creative_seed,
            mc_infotext.CREATIVE_LLM_SEED: self.roll.llm_seed,
            # What the user typed, brackets and all -- not the cleaned text the
            # writer saw. §15: a restore is supposed to hand back the prompt
            # they wrote, and a source phrase with its literal commands quietly
            # missing would be a workflow that cannot be continued from.
            mc_infotext.CREATIVE_SOURCE: self.roll.typed,
        }
        compact = getattr(recipe, "compact", "")
        if compact:
            recorded[mc_infotext.CREATIVE_RECIPE] = compact
        version = getattr(recipe, "library_version", "")
        if version:
            recorded[mc_infotext.CREATIVE_LIBRARY] = version
        axes = mc_infotext.creative_axes(self.settings.get("axis_modes"),
                                         self.settings.get("fixed_values"))
        if axes:
            recorded[mc_infotext.CREATIVE_AXES] = axes
        excluded = mc_infotext.creative_exclusions(self.settings.get("excluded_values"))
        if excluded:
            recorded[mc_infotext.CREATIVE_EXCLUDED] = excluded
        if "anti_repetition" in self.settings:
            recorded[mc_infotext.CREATIVE_ANTI] = str(
                bool(self.settings.get("anti_repetition")))
        writer = writer_identity()
        if writer:
            recorded[mc_infotext.CREATIVE_WRITER] = writer
        recorded.update(self.spatial or {})
        recorded.update(self.literal_metadata)
        return recorded

    @property
    def literal_metadata(self) -> dict:
        """What this generation records about its literal commands: how many.

        The count and the syntax version, and deliberately not the payloads. The
        payloads are already in the image's own ``Prompt:`` line, which is where
        a reader looks for what the model was given, and in the recorded source
        prompt with their brackets on, which is where a restore looks for what
        the user wrote. A third copy under a key of ours would be a few hundred
        bytes repeating the file, and a copy a later paste could disagree with.

        Absent entirely when there were none, like every other optional key
        here: an ordinary image should say nothing about a feature it did not
        use.
        """
        import mc_infotext
        from prompt_master.krea import literals

        count = len(getattr(self.literals, "commands", ()) or ())
        if not count:
            return {}
        return {mc_infotext.LITERAL_VERSION: literals.SYNTAX_VERSION,
                mc_infotext.LITERAL_COUNT: count}


def prepare(roll: Roll | None, stored=None, prompt=None, spatial=None,
            inheritable=None, literals=None) -> Prepared:
    """One finished prompt, packaged for the processing hook.

    ``prompt`` is what Stage 1 generates from, with the literal commands already
    restored into it: the structured Spatial BBOX prompt when there was a
    layout, and the writer's own paragraph when there was not. It is assembled
    by the caller because only the caller knows which of those two it is, and
    because restoring the literals exactly once means doing it in exactly one
    place per generation.

    ``inheritable`` is the same prompt with no literal payload in it, for
    Stage 2. It defaults to ``prompt`` -- a generation with no literals has
    nothing to isolate, and the two really are the same string then.

    ``roll`` is ``None`` for a Spatial-only generation and for a generation
    whose only change was its literals, where no writer ran and ``prompt`` is
    the whole of what is being substituted.
    """
    body = prompt
    if body is None:
        body = roll.expanded if roll is not None else ""
    body = str(body or "").strip()
    if inheritable is None:
        inheritable = body
    return Prepared(roll=roll,
                    generation=body,
                    inheritable=str(inheritable or "").strip(),
                    literals=literals,
                    settings=dict(stored or {}),
                    spatial=dict(spatial or {}))


def writer_identity() -> str:
    """Which language model wrote the prompt, as far as this installation knows.

    The file's own name, plus the catalogue id when the backbone came from the
    catalogue -- because two people running "gemma-3-12b-it" may not be running
    the same weights, and the managed id is the one string that says which
    curated build it was. Not a hash: hashing a twelve-gigabyte file to label a
    PNG is a cost paid on every generation to answer a question almost nobody
    asks, and the answer would still be absent for a manual install of a file
    that has since been replaced in place.

    Empty when nothing can be determined, which keeps the key out of the
    infotext entirely rather than recording a guess.
    """
    try:
        import mc_llm_runtime

        config = mc_llm_runtime.config()
    except Exception:
        logger.debug("Model Chain: could not identify the Krea writer", exc_info=True)
        return ""

    model = getattr(config, "model", None)
    name = model.stem if model is not None else ""
    managed = str(getattr(config, "managed_id", "") or "")
    if managed and managed != name:
        return f"{name} ({managed})".strip() if name else managed
    return name


# --------------------------------------------------------------------------- #
# The checkpoint guard
# --------------------------------------------------------------------------- #


def checkpoint_objection() -> str:
    """Why Creative Mode must not arm against the selected checkpoint, or "".

    Creative Mode writes Krea 2 prompts: long, natural-language, written to
    Krea's own guidance. Handing one to SD 1.5 is not a smaller version of the
    feature, it is a paragraph fed to a model with 77 tokens of room, and the
    result would look like the extension was broken rather than like a choice.

    An architecture that cannot be identified is allowed through without
    complaint. Detection reads a checkpoint header and genuinely cannot see
    inside every GGUF or repacked build, so refusing on "unknown" would refuse
    real Krea 2 checkpoints -- and a guard that blocks the thing it exists to
    protect is worse than no guard.

    This is a txt2img concern only. LLM Studio writes prompts and generates no
    images, so which checkpoint happens to be loaded there is none of its
    business.
    """
    try:
        import mc_arch
        from modules import shared

        found = mc_arch.detect_loaded_engine()
        if found is mc_arch.UNKNOWN:
            found = mc_arch.detect_from_checkpoint_name(shared.opts.sd_model_checkpoint)
    except Exception:
        logger.debug("Model Chain: could not identify the image checkpoint for Creative Mode",
                     exc_info=True)
        return ""

    if found is mc_arch.UNKNOWN or found.key == "krea2":
        return ""
    return (f"Creative Mode writes Krea 2 prompts, and the selected checkpoint is "
            f"{found.label}. Select a Krea 2 checkpoint, or turn Creative Mode off to "
            "generate with the prompt as you typed it.")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

ENABLED = "krea_creative_enabled"
CREATIVITY = "krea_creativity"
SEED = "krea_creative_seed"
ANTI_REPETITION = "krea_creative_anti_repetition"
AXIS_MODES = "krea_creative_axis_modes"
FIXED_VALUES = "krea_creative_fixed"
EXCLUDED_VALUES = "krea_creative_excluded"
DIRECTIONS = "krea_creative_directions"
"""The axes the user has added a direction for, complete or not.

Natural is the absence of a decision, which the three keys above express
perfectly and cannot express *twice*: an axis nobody has directed and an axis
somebody added a moment ago and has not yet chosen treatments for are both
``natural`` underneath, and must stay that way -- the Director has to ignore
both, and the second one is not a half-configured direction it should guess at.

They differ only on screen, where the second one is a row waiting to be filled
in. That is a fact about the panel, so the panel keeps it here rather than
encoding it as a mode the generation would then have to know about.
"""
HISTORY = "krea_creative_history"
PROFILE = "krea_creative_profile"
"""Which named profile the settings above were last loaded from, or "".

Remembered so the panel can open showing where the configuration came from. It
is a *label on the settings*, not a source of them: nothing applies a profile
because this says so, which is what keeps opening a tab from silently discarding
whatever was adjusted in the last one.
"""

CONFIGURATION = (CREATIVITY, SEED, ANTI_REPETITION, AXIS_MODES, FIXED_VALUES,
                 EXCLUDED_VALUES, DIRECTIONS)
"""The keys a Creative profile describes, and the ones it deliberately does not.

:data:`ENABLED` is absent and stays absent. A profile says *how* Creative Mode
behaves; whether the feature is on is a decision somebody makes at the moment
they press Generate, and a preset that could switch it on would be a preset that
changes what the button does. :data:`HISTORY` is absent too -- it is a record of
what recently happened on this machine, not a setting.
"""


def settings() -> dict:
    """Every Creative Mode preference, with the package's defaults filled in.

    One set of settings for both surfaces, not two. The axes, the Creativity
    position and the seed describe *how this installation does art direction*,
    and a user who has spent five minutes configuring ten axes in LLM Studio
    should not have to do it again in txt2img. The library's own ``defaults.json``
    supplies anything the file has never held.
    """
    from prompt_master.krea import director, variation

    try:
        stored = mc_llm_state.preferences()
    except Exception:
        logger.debug("Model Chain: could not read the Creative Mode preferences",
                     exc_info=True)
        stored = {}

    try:
        from prompt_master.krea import library as library_module

        defaults = dict(library_module.library().defaults)
        axis_keys = library_module.library().axis_keys
    except Exception:
        logger.debug("Model Chain: the creativity library could not be read", exc_info=True)
        defaults, axis_keys = {}, ()

    modes = dict(defaults.get("axis_modes") or {})
    modes.update({key: str(value).casefold()
                  for key, value in (stored.get(AXIS_MODES) or {}).items()
                  if str(value).casefold() in director.MODES})
    fixed = dict(defaults.get("fixed_values") or {})
    fixed.update({key: str(value) for key, value in (stored.get(FIXED_VALUES) or {}).items()
                  if value})
    excluded = dict(defaults.get("excluded_values") or {})
    excluded.update({key: value for key, value in (stored.get(EXCLUDED_VALUES) or {}).items()
                     if value})

    return {
        "enabled": bool(stored.get(ENABLED, defaults.get("creative_mode_enabled", False))),
        "creativity": variation.clamp(stored.get(CREATIVITY,
                                                 defaults.get("creativity", variation.DEFAULT))),
        "seed": _seed(stored.get(SEED, defaults.get("creative_seed", director.RANDOM_SEED))),
        "anti_repetition": bool(stored.get(ANTI_REPETITION,
                                           defaults.get("anti_repetition", True))),
        # Natural for anything nobody has set. A fresh install has no art
        # direction in it at all, and an axis a later package adds arrives
        # silent rather than quietly varying.
        "axis_modes": {key: modes.get(key, director.NATURAL) for key in axis_keys},
        "fixed_values": known_fixed(fixed),
        "excluded_values": known_excluded(excluded),
        "directions": known_directions(stored.get(DIRECTIONS), modes, axis_keys),
    }


def known_directions(chosen, modes, axis_keys) -> list[str]:
    """The axes with a row on the panel, in the library's own order.

    Every axis that is actually directing has a row whether or not this key
    mentions it. That is what makes the key additive rather than a second
    source of truth: a settings file or a profile written before the treatment
    picker existed has modes and no list, and its directions still appear.
    """
    from prompt_master.krea import director

    keys = list(axis_keys)
    named = {str(key) for key in (chosen or ()) if str(key) in keys}
    directing = {key for key in keys
                 if modes.get(key) in (director.VARY, director.FIXED)}
    return [key for key in keys if key in named or key in directing]


def known_fixed(fixed) -> dict:
    """Pinned ids the current library still has, by axis.

    A saved configuration outlives the package version it was written against.
    An id that has gone is dropped here, once, with a line in the log naming it
    -- rather than reaching the Director, which would fall silent on that axis
    every roll without anybody being told why.
    """
    axes = _axes()
    if axes is None:
        return {key: str(value) for key, value in (fixed or {}).items() if value}

    kept, lost = {}, []
    for key, value in (fixed or {}).items():
        axis = axes.get(key)
        if axis is None or not value:
            continue
        if axis.variant(str(value)) is None:
            lost.append(f"{key}={value}")
            continue
        kept[key] = str(value)
    if lost:
        logger.warning("Model Chain: pinned Creative treatments are not in this "
                       "creativity library and were dropped: %s", ", ".join(lost))
    return kept


def known_excluded(excluded) -> dict:
    """Excluded ids the current library still has, by axis.

    Same rule as the pinned ones and for a sharper reason: an exclusion naming a
    treatment that no longer exists excludes nothing, and keeping it would leave
    a user reading a list of protections they no longer have.
    """
    axes = _axes()
    kept, lost = {}, []
    for key, values in (excluded or {}).items():
        if isinstance(values, str):
            values = [values]
        chosen = []
        for value in values or ():
            identifier = str(value).strip()
            if not identifier:
                continue
            axis = axes.get(key) if axes is not None else None
            if axes is not None and (axis is None or axis.variant(identifier) is None):
                lost.append(f"{key}={identifier}")
                continue
            if identifier not in chosen:
                chosen.append(identifier)
        if chosen:
            kept[key] = chosen
    if lost:
        logger.warning("Model Chain: excluded Creative treatments are not in this "
                       "creativity library and were dropped: %s", ", ".join(lost))
    return kept


def _axes():
    """``{axis key: Axis}`` for the loaded library, or ``None`` if it will not load.

    ``None`` rather than an empty mapping, because the two mean opposite things
    here: no library is a reason to keep every stored id untouched, and an empty
    library would be a reason to throw them all away.
    """
    try:
        from prompt_master.krea import library as library_module

        lib = library_module.library()
    except Exception:
        logger.debug("Model Chain: the creativity library could not be read", exc_info=True)
        return None
    return {key: lib.axis(key) for key in lib.axis_keys}


def _seed(value) -> int:
    from prompt_master.krea import director

    try:
        return int(value)
    except (TypeError, ValueError):
        return director.RANDOM_SEED


def remember(**values) -> None:
    """Keep a Creative Mode preference. Never fatal: this is a convenience."""
    try:
        mc_llm_state.remember(**values)
    except Exception:
        logger.debug("Model Chain: could not save the Creative Mode preferences",
                     exc_info=True)


def axis_settings(stored=None) -> dict:
    """The stored modes, pins and exclusions as the Director's own type."""
    from prompt_master.krea import director

    stored = stored or settings()
    fixed = stored.get("fixed_values") or {}
    excluded = stored.get("excluded_values") or {}
    return {key: director.AxisSetting(mode=mode, fixed_id=fixed.get(key),
                                      excluded_ids=frozenset(excluded.get(key) or ()))
            for key, mode in (stored.get("axis_modes") or {}).items()}


def active_axes(stored=None) -> list[str]:
    """The axes this configuration actually directs, in the library's own order.

    The list the panel draws, and the shortest true description of what Creative
    Mode is doing: everything else is Natural, and Natural is absence. An axis
    whose mode is Fixed with nothing pinned is still active -- the user asked for
    a pin and has not chosen one yet, and hiding the row would hide the half-made
    decision rather than the decision.
    """
    from prompt_master.krea import director

    stored = stored or settings()
    modes = stored.get("axis_modes") or {}
    keys = list(modes)
    axes = _axes()
    if axes is not None:
        keys = [key for key in axes if key in modes]
    return [key for key in keys if modes.get(key) in (director.VARY, director.FIXED)]


# --------------------------------------------------------------------------- #
# What an image says about how it was made
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Setup:
    """One image's Creative configuration, read back out of its infotext.

    Not applied by reading it. This is what the *ordinary* paste deliberately
    does not do: a paste restores the picture, which means restoring the final
    expanded prompt and switching Creative Mode off so nothing expands it twice.
    Everything here is for the second, explicit action -- "restore the creative
    setup" -- which puts the short source phrase back in the prompt box and the
    configuration back on the panel, and says so.
    """

    source: str = ""
    creativity: int | None = None
    seed: int | None = None
    llm_seed: int | None = None
    recipe: str = ""
    library_version: str = ""
    axis_modes: dict = field(default_factory=dict)
    fixed_values: dict = field(default_factory=dict)
    excluded_values: dict = field(default_factory=dict)
    anti_repetition: bool | None = None
    loras: str = ""
    """What an older image recorded in its Pinned LoRAs field, if it had one.

    Read and shown, never applied. The control it came from is gone and
    ``[[<lora:name:weight>]]`` in the prompt is what replaced it, so restoring
    this into a setting that no longer exists would restore nothing; showing it
    is how somebody continuing from an old image can see which tags that picture
    used and type them back where they now belong.
    """

    literal_positive: str = ""
    literal_negative: str = ""
    """What the two Literal Prompt boxes held when this image was made.

    Read and shown like everything else here, and -- unlike :attr:`loras` --
    restorable, because the controls they came from still exist. The explicit
    "Restore Creative setup" button puts them back; an ordinary paste clears
    them instead, because the recorded prompt already contains these payloads
    and a paste that refilled the boxes would insert them a second time.
    """

    writer: str = ""

    spatial_layout: str = ""
    spatial_compose_mode: str = ""
    spatial_version: int | None = None
    """What the image recorded about its Spatial Layout, if it had one.

    Carried on the Creative record rather than in a second one, because there is
    one restore button and one thing it restores: the workflow that made this
    picture. A spatial image is a Creative image with a canvas, and splitting the
    record would mean two buttons whose only difference is which half of the same
    workflow they put back.
    """

    @property
    def present(self) -> bool:
        """Whether this describes a Creative generation at all."""
        return bool(self.source or self.recipe or self.creativity is not None
                    or self.spatial_layout)

    @property
    def spatial(self) -> bool:
        """Whether this image recorded a spatial layout worth restoring."""
        return bool(self.spatial_layout)

    @property
    def literals(self) -> bool:
        """Whether this image recorded Literal Prompt fields worth restoring.

        Deliberately not folded into :attr:`present`. An image whose only
        Model Chain record is two literal fields is not a Creative image, and
        saying "Creative image restored" over it would be describing a feature
        that never ran.
        """
        return bool(self.literal_positive or self.literal_negative)

    @property
    def recorded(self) -> bool:
        """Whether this image recorded anything worth offering to restore.

        Wider than :attr:`present`, and the two are wanted apart. ``present``
        answers "is this a Creative image", which is what decides whether a
        paste says so and whether restoring turns the writer back on. This
        answers "is there anything here at all", which is what decides whether
        the record is kept -- and a pair of Literal Prompt boxes with both
        features switched off is a whole restorable setup that is not a
        Creative one.
        """
        return bool(self.present or self.literals)

    @property
    def replayable(self) -> bool:
        """Whether the recorded recipe is enough to reproduce the art direction."""
        return bool(self.recipe)

    def warnings(self) -> list[str]:
        """What about this record the current installation cannot honour.

        Said before anything is restored rather than discovered afterwards. A
        different library version does not stop a replay -- ids are stable by the
        package's own contract -- but it does mean the *expressions* those ids
        render into may have been rewritten, and somebody comparing two pictures
        deserves to know that before they conclude the feature is broken.
        """
        found = []
        version = installed_library_version()
        if self.library_version and version and self.library_version != version:
            found.append(f"This image was made with creativity library "
                         f"{self.library_version}; {version} is installed. The recorded "
                         "treatments still resolve, but their wording may have changed.")
        writer = writer_identity()
        if self.writer and writer and self.writer != writer:
            found.append(f"The prompt was written by {self.writer}; {writer} is "
                         "configured now. The same brief will not produce the same "
                         "paragraph.")
        if self.spatial_layout:
            from prompt_master.krea import spatial as spatial_module

            if self.spatial_version is not None \
                    and self.spatial_version != spatial_module.VERSION:
                found.append(f"This image records spatial layout version "
                             f"{self.spatial_version}; this build reads version "
                             f"{spatial_module.VERSION}. The canvas was not restored.")
            else:
                parsed = spatial_module.parse(self.spatial_layout)
                found.extend(parsed.notes)
        return found


def installed_library_version() -> str:
    try:
        from prompt_master.krea import library as library_module

        return library_module.library().version
    except Exception:
        return ""


class Pasted:
    """The Creative setup from the most recent infotext paste, if there was one.

    One slot, overwritten by each paste, read by the button that restores it.
    Deliberately not a queue and deliberately not consumed on read: somebody who
    pastes an image, looks at what it says, and then decides to continue from it
    should find it still there, and somebody who pastes a second image should
    find the second one rather than a backlog.

    It holds a parsed record and never a prompt to be generated from. Nothing
    here can reach a generation on its own -- the only way anything in it is
    applied is a button press.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._setup: Setup | None = None

    def remember(self, setup: Setup | None) -> None:
        with self._lock:
            self._setup = setup if setup is not None and setup.recorded else None

    @property
    def setup(self) -> Setup | None:
        with self._lock:
            return self._setup

    def clear(self) -> None:
        with self._lock:
            self._setup = None


@dataclass(frozen=True)
class ReplayPlan:
    """Art direction to reuse verbatim on the next roll, instead of drawing one."""

    creativity: int
    creative_seed: int
    llm_seed: int
    recipe: str
    library_version: str = ""
    source: str = ""

    @property
    def description(self) -> str:
        counted = len(self.recipe.split(",")) if self.recipe else 0
        return (f"{counted} recorded {'line' if counted == 1 else 'lines'} of art "
                f"direction at Creativity {self.creativity}, creative seed "
                f"{self.creative_seed}")


class Replay:
    """One armed replay, or none. Explicit to arm, spent by the next roll.

    This is not the arming token that used to sit between a Gradio handler and a
    native Generate, and the difference is the whole reason it is allowed to
    exist. That token held a *finished prompt* -- a model's output, made before
    the click, waiting for a generation to spend it -- which is how a closed tab
    used to strand work and how the wrong image could have picked it up.

    What is held here is a list of variant ids the user asked to reuse: data they
    chose, that they can read on the panel before pressing anything, that no
    model produced and that reaches nothing on its own. The roll still happens
    inside the generation and the writer is still called exactly once. All this
    changes is where that roll's art direction comes from, and it says so on the
    panel for as long as it is armed.

    One generation, then gone: :meth:`take` clears it. A replay that survived its
    generation would be a setting nobody set, quietly reproducing an old picture's
    direction over every new idea typed after it.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._plan: ReplayPlan | None = None

    def arm(self, plan: ReplayPlan | None) -> ReplayPlan | None:
        with self._lock:
            self._plan = plan
            return self._plan

    def take(self) -> ReplayPlan | None:
        """The armed plan, and disarm. Called by the roll, once."""
        with self._lock:
            plan, self._plan = self._plan, None
            return plan

    @property
    def pending(self) -> ReplayPlan | None:
        with self._lock:
            return self._plan

    def clear(self) -> None:
        self.arm(None)


pasted = Pasted()
"""What the last paste said about Creative Mode. Read by the restore button."""

replay = Replay()
"""The armed replay, if the user asked for one. Spent by the next roll."""


# --------------------------------------------------------------------------- #
# What the last few rolls used
# --------------------------------------------------------------------------- #


def history() -> list[list[str]]:
    """The variant ids of the last few rolls, newest last.

    Ids and not prompts. The design is explicit about this and it is worth
    keeping explicit in the code: what anti-repetition needs to know is "did we
    just use the impasto medium", and storing the prompts to answer that would
    be keeping a transcript of everything anybody asked for in a preferences
    file.
    """
    try:
        stored = mc_llm_state.preferences().get(HISTORY) or []
    except Exception:
        return []
    rolls = []
    for entry in stored:
        if isinstance(entry, (list, tuple)):
            rolls.append([str(identifier) for identifier in entry])
    return rolls


def recent_ids() -> tuple[str, ...]:
    """Every id in the remembered history, flattened, for the Director."""
    return tuple(identifier for roll in history() for identifier in roll)


def note_roll(recipe) -> None:
    """Remember what one roll used, and forget the oldest one past the limit."""
    identifiers = [item.variant_id for item in getattr(recipe, "items", ())]
    if not identifiers:
        return
    try:
        from prompt_master.krea import library as library_module

        limit = library_module.library().anti_repetition.history_length
    except Exception:
        limit = 8
    remember(**{HISTORY: (history() + [identifiers])[-limit:]})


def forget_history() -> None:
    """Drop the recent-roll memory, so the next roll may pick anything again."""
    remember(**{HISTORY: []})


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class Creative:
    """Creative Mode's state for one WebUI. One field and a lock.

    Compare the module this replaced, which had a cache, a revision counter, a
    cancellation handle, a failure count and a status machine. All of those
    existed to manage work nobody had asked for. What is left is the last roll,
    so the diagnostic view has something to show -- and it is the *only* thing
    left, because there is no longer a gap between a roll and the generation
    that uses it for state to live in.

    There used to be an arming token as well. It was permission for exactly one
    native generation to spend a roll that had been made ahead of the click, in
    a Gradio handler, and it existed because that gap existed: between the roll
    finishing and the browser clicking Generate, the result had to be somewhere,
    and it had to be impossible for a second or queued generation to pick it up.
    The roll runs inside the generation now. The gap is gone, and so is
    everything that guarded it.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._last: Roll | None = None
        self._status = IDLE

    @property
    def status(self) -> str:
        return self._status

    def say(self, status: str) -> str:
        self._status = status
        return status

    @property
    def last(self) -> Roll | None:
        with self._lock:
            return self._last

    # -- one roll ---------------------------------------------------------- #

    def roll(self, source: str, stored=None, references=(), guard_checkpoint=False,
             task_id="", own_bar: bool = True, spatial_layout=None, raw_source=""):
        """One creative roll: direct locally, then ask the model once.

        Yields :class:`mc_llm_sessions.Event` throughout so a caller can put the
        writer's progress on screen, and finishes with ``DONE`` carrying the
        expanded prompt. The recipe is on :attr:`last` by then.

        The Director runs first and runs entirely in this process. By the time
        the writer is called, every creative decision has already been made and
        written down; the model's whole job is to turn one brief into one Krea
        prompt. That ordering is what makes "exactly one model call" a property
        of the design rather than a thing to be careful about.

        ``task_id`` is a host progress task to report on, phase by phase, which
        is the only thing standing between a user and twenty seconds of a screen
        that looks broken. ``own_bar`` says whether that task is this roll's to
        start and finish. It is false for the txt2img path, where the roll runs
        inside a generation whose bar the host started and must go on running
        after the roll ends; there ``task_id`` may be empty as well, because the
        bar being borrowed already exists whatever it is called. See
        :mod:`mc_llm_progress`.

        ``spatial_layout`` is a parsed :class:`prompt_master.krea.spatial.Layout`
        when the user has drawn one, and it reaches the writer as **one
        sentence** and nothing else: placement is handled separately, so do not
        state rigid screen positions. Not the regions, not their prompts, not a
        coordinate. Handing pass 1 the region text would put the user's own
        words through a rewriter and then let the compositor place the original
        beside the rewrite, which is a request for two of the same subject.

        With no layout -- which is every generation until somebody draws one --
        the user turn is assembled from ``recipe.brief`` exactly as it always
        was, down to the byte, so llama.cpp resumes at the same prefix and
        Creativity 1 stays the compatibility guarantee it is documented as.

        ``source`` is the *transformable* text and the caller has already lifted
        the literal commands out of it. That is the whole boundary this feature
        draws, and it is drawn one layer up rather than here for a reason worth
        stating: the Director reads the source too -- to notice that the user
        said "oil painting" and lock the Medium axis -- and it must not lock an
        axis because the word "anime" appears inside a LoRA filename. Both
        passes read one string, and neither of them can see a payload.

        ``raw_source`` is what the user typed, kept on the :class:`Roll` so the
        metadata can record it. Nothing in this method reads it.
        """
        from prompt_master.core.models import draw_seed
        from prompt_master.krea import director

        source = str(source or "").strip()
        if not source:
            yield sessions.Event(sessions.FAILED,
                                 "Type what you want in the prompt box first.")
            return

        if guard_checkpoint:
            objection = checkpoint_objection()
            if objection:
                yield sessions.Event(sessions.FAILED, objection)
                return

        stored = stored or settings()
        self.say(DIRECTING)
        # Taken before the try, and taken whatever happens next: an armed replay
        # is permission for *this* roll, and one that survived a failed roll
        # would silently direct the next unrelated press.
        plan = replay.take()
        try:
            if plan is not None:
                recipe = director.replay(
                    source=source, creativity=plan.creativity,
                    creative_seed=plan.creative_seed, llm_seed=plan.llm_seed,
                    recipe_ids=plan.recipe)
            else:
                recipe = director.roll(
                    source=source,
                    creativity=stored["creativity"],
                    creative_seed=stored["seed"],
                    settings=axis_settings(stored),
                    history=recent_ids() if stored.get("anti_repetition") else ())
        except Exception as exc:
            # A library that will not load is a Creative Mode that cannot
            # direct. It is emphatically not a reason to send the request
            # anyway with no brief: the user asked for art direction, and a
            # plain expansion pretending to be one is the wrong answer.
            self.say(ERROR)
            logger.debug("Model Chain: the Creative Director failed", exc_info=True)
            yield sessions.Event(sessions.FAILED, f"The creativity library could not be "
                                                  f"used: {exc}")
            return

        for note in getattr(recipe, "notes", ()):
            # Warned rather than logged quietly: every note here is a line of art
            # direction the user configured and did not get, and the whole reason
            # exclusions are absolute is that guessing past them would be worse.
            logger.warning("Model Chain: Creative Mode — %s", note)
            yield sessions.Event(sessions.STATUS, note)

        yield sessions.Event(sessions.STATUS, _directed(recipe))
        self.say(WRITING)

        cancel = sessions.Cancellation()
        # Held by name and closed explicitly: the loop below stops as soon as it
        # has the finished prompt, and the statement that gives the GPU back is
        # in that run's ``finally``. Closing it is what runs that finally now
        # rather than whenever the interpreter next collects the frame.
        # ``guard_checkpoint`` is the txt2img path and only the txt2img path, so
        # it is also exactly the condition under which an image generation
        # follows this roll -- which makes it the right thing to key the VRAM
        # reserve on. LLM Studio writes a prompt and stops; reserving image VRAM
        # there would shrink the writer for a picture nobody asked for.
        reserve = image_reserve_bytes() if guard_checkpoint else 0
        direction = recipe.brief
        if spatial_layout is not None and getattr(spatial_layout, "regions", ()):
            from prompt_master.krea import spatial as spatial_module

            direction = spatial_module.directed(direction)
        run = sessions.krea(source, list(references or []), recipe.llm_seed, cancel,
                            recipe.creativity, direction, reserve)
        progress = mc_llm_progress.reporter
        warm = _warm()
        progress.begin(task_id,
                       _prompt_size(source, references, direction, cached=warm),
                       warm, claim=own_bar)
        written = ""
        finished = False
        try:
            for event in run:
                if event.kind == sessions.DONE:
                    written = event.text
                    finished = True
                    break
                if event.kind in (sessions.FAILED, sessions.CANCELLED):
                    self.say(ERROR if event.kind == sessions.FAILED else IDLE)
                    yield event
                    return
                if event.kind == sessions.CHUNK:
                    # The first chunk is the moment prompt evaluation ended and
                    # generation began -- the one boundary in the whole run that
                    # is observable from here, and the one the bar most needs.
                    progress.enter(mc_progress.PHASE_KREA_WRITE)
                    progress.wrote(event.text)
                elif event.kind == sessions.STATUS:
                    progress.enter(_phase_for(event.text))
                if progress.interrupted():
                    cancel.cancel()
                    self.say(IDLE)
                    yield sessions.Event(sessions.CANCELLED, "Stopped.")
                    return
                yield event
        except Exception as exc:
            self.say(ERROR)
            logger.debug("Model Chain: the Creative Mode roll failed", exc_info=True)
            yield sessions.Event(sessions.FAILED, str(exc))
            return
        finally:
            run.close()
            if finished:
                progress.end(written)
            else:
                progress.abandon()

        if not written.strip():
            self.say(ERROR)
            yield sessions.Event(sessions.FAILED, "The model returned an empty prompt.")
            return

        with self._lock:
            self._last = Roll(recipe=recipe, source=source, expanded=written.strip(),
                              raw_source=str(raw_source or ""))
        # A replay is not a new choice and does not go in the recent memory. The
        # ids it used were already recorded when they were first drawn, and
        # writing them again would push a user's own reproduction away from the
        # treatments they had just asked to reproduce.
        if stored.get("anti_repetition") and not getattr(recipe, "replayed", False):
            note_roll(recipe)

        # The last thing before the prompt is handed over, and only on the path
        # that is about to generate an image. The reserve above should mean
        # there is nothing to do; this is what recovers a card that was already
        # full when the roll started.
        #
        # Not when a Spatial Composer is still to run, though, and that
        # exception is the whole of section 13. Handing the card back here asks
        # the broker for image VRAM, and the only way the broker can give it is
        # by stopping llama-server -- the same llama-server the Composer is
        # about to send its one request to, a fraction of a second later. What
        # that costs is a second GGUF load and a cold prompt cache in the
        # middle of one generation, to free VRAM that nothing will use until
        # the Composer has finished with the card anyway.
        #
        # The rule generalises: do not reclaim a persistent service between two
        # consecutive operations that both need it. The hand-back happens after
        # the last LLM phase instead, which the caller performs.
        if guard_checkpoint and not _composer_follows(spatial_layout):
            freed = hand_back_vram()
            if freed:
                logger.info("Model Chain: freed %.1f GB for the image generation that "
                            "follows the Krea roll", freed / (1024 ** 3))
                yield sessions.Event(sessions.STATUS,
                                     "Handing the card back for the image…")

        self.say(READY)
        yield sessions.Event(sessions.DONE, written.strip())


def _prompt_size(source: str, references, brief: str, cached: bool = False) -> int:
    """How much text the model is about to *evaluate*, in characters.

    Not how much it is about to be sent. Those are different numbers and the
    difference is most of the request: llama.cpp keeps a prompt cache, Krea's
    instruction is the same bytes on every roll, and a server that has already
    answered one has it. What is left to evaluate is the user's line and the
    creative brief -- and the brief is different every roll by construction, so
    it can never be cached, which is what makes it the whole cost.

    Measured on one user's ``llama-server.log``, mid-run: 1,028 prompt tokens, of
    which 646 came out of the cache and 382 were evaluated, at 27 ms each. Ten
    and a half seconds, all of it the brief.

    Counting the instruction anyway -- which this did -- made two things wrong at
    once. The bar over-predicted the reading phase for a short brief, because it
    was pricing two kilobytes that cost nothing; and the rate it learned per
    character drifted with the mix, since the same seconds were being divided by
    a character count that included a constant. Sizing by what is actually read
    fixes both, and makes ``krea:read`` mean one thing.

    ``cached`` is "a server is already up and has answered a Krea request the
    same way". It guesses false when nothing can be told, which prices the
    instruction in -- the pessimistic direction, and the correct one for a cold
    start, where it really is read.
    """
    from prompt_master.krea import enhancer

    try:
        size = len(enhancer.user_content(source, None, brief))
        if not cached:
            size += len(enhancer.system_prompt(bool(references)))
    except Exception:
        logger.debug("Model Chain: could not size the Krea prompt", exc_info=True)
        return max(len(source or "") + len(brief or ""), 1)
    return max(size, 1)


def _composer_follows(spatial_layout) -> bool:
    """Whether a Spatial Composer call comes after this roll on the same server.

    ``True`` only for a Smart layout that actually has boxes in it. Spatial
    Layout switched on over an empty canvas composes nothing, and a Direct
    merge is deterministic and asks no model anything -- in both of those the
    roll really is the last LLM phase of the plan, and the card may go back to
    the image side immediately.
    """
    if spatial_layout is None or not getattr(spatial_layout, "regions", ()):
        return False
    try:
        from prompt_master.krea import spatial as spatial_module

        return getattr(spatial_layout, "compose_mode", "") == spatial_module.SMART
    except Exception:
        logger.debug("Model Chain: could not tell whether a Composer follows this roll",
                     exc_info=True)
        return False


def image_reserve_bytes() -> int:
    """VRAM to keep clear for the image generation this roll is about to trigger.

    Creative Mode inverted an order that used to be safe. Before it, the image
    checkpoint was loaded first and the language model negotiated its placement
    against whatever was left -- which is why an LLM that had to squeeze into
    three gigabytes did so, and why both fitted. A Creative roll loads the
    language model *first*, onto a card with nothing on it, so llama.cpp sizes
    itself to the whole thing and the checkpoint that has to run three hundred
    milliseconds later gets the remainder. On a 24 GB card that is the
    difference between "both fit" and "the image model does not".

    So the roll says up front how much room to leave. ``negotiate`` already
    takes exactly this number and works hard to honour it by shrinking context
    or offloading blocks; leaving the room is very much cheaper than reclaiming
    it afterwards, because a running llama-server can only give VRAM back by
    stopping.

    What is already the image family's is subtracted, and it is subtracted from
    :func:`mc_broker.held_bytes` rather than the residency register. The
    register only holds what was *declared* to it, and image checkpoints never
    are -- Forge loads and moves them, and ``mc_memory`` cooperates with that
    rather than announcing every load to the broker. Asking the register
    therefore answered 0 for a checkpoint sitting in fourteen gigabytes of
    VRAM, and this function reserved room all over again for a model that was
    already there. What that cost, on a 24 GB card holding a 13.9 GB
    checkpoint, was a language model sized against 8.7 GB *minus* another
    13.9 GB: sixteen of forty-eight blocks on the card and the rest crawling
    from system RAM, on a machine with room for far more than that.

    What this function answers on its own is a Stage-1-shaped question, and a
    long chain is not Stage-1-shaped. Krea 2 into Klein 9B at a 1.5x multiplier
    has its largest moment in the *handoff*, where Stage 2's weights and Stage
    1's spared encoders are on the card together, and a language model placed
    against Stage 1's requirement alone is holding VRAM that phase needs. So
    when a plan has been published, the plan answers -- it knows about Stage 2,
    the transition and the warm-up, and it takes the largest of them rather
    than the first. See :func:`mc_plan.llm_reserve_bytes`.

    Zero on any failure, and zero is the old behaviour: an unknown checkpoint
    is not a reason to refuse to write a prompt.
    """
    try:
        import mc_plan

        if mc_plan.current() is not None:
            return mc_plan.llm_reserve_bytes()
    except Exception:
        logger.debug("Model Chain: could not size the reserve from the active plan",
                     exc_info=True)

    try:
        import mc_broker
        import mc_memory
        from modules import shared

        name = str(getattr(shared.opts, "sd_model_checkpoint", "") or "")
        if not name:
            return 0
        required = int(mc_memory.vram_required_bytes(name))
        held = int(mc_broker.held_bytes(mc_broker.FAMILY_IMAGE))
        # The global safety margin comes off too, because ``negotiate`` adds it
        # on top of whatever this returns and it is the *same* number: both are
        # ``mc_memory.vram_headroom_bytes()``, which is inside
        # ``vram_required_bytes`` already. Counted twice it is a gigabyte and a
        # half of a card reserved for one activation peak, which on a 24 GB
        # machine is several blocks of the language model for nothing.
        margin = int(mc_broker.safety_margin_bytes())
        return max(required - max(held, 0) - max(margin, 0), 0)
    except Exception:
        logger.debug("Model Chain: could not size the image reserve for a Creative roll",
                     exc_info=True)
        return 0


def hand_back_vram(reason: str = "the image generation that follows a Krea roll") -> int:
    """Ask for the room the coming image pass needs, if something else holds it.

    The reserve above prevents the problem; this recovers from it. A
    llama-server that was already up when the reserve was introduced -- or that
    was placed for a different checkpoint, or before the user changed the image
    model -- is holding VRAM nobody can shrink in place, and the only way to get
    it back is to ask the broker, which stops the server.

    It is a no-op in the ordinary case: ``request_vram`` returns immediately
    when what is free already covers the requirement, so a card that was sized
    correctly by the reserve never pays for this call.
    """
    needed = image_reserve_bytes()
    if needed <= 0:
        return 0
    try:
        import mc_broker

        return mc_broker.request_vram(mc_broker.FAMILY_IMAGE, needed, reason=reason).freed
    except Exception:
        logger.debug("Model Chain: could not ask for image VRAM after a Creative roll",
                     exc_info=True)
        return 0


def _warm() -> bool:
    """Whether llama-server is already up, for the waiting phase's prediction.

    A wrong answer costs one roll a poor ETA on the phase that is over before
    anybody reads it, so this never raises and guesses cold -- the pessimistic
    direction -- when it cannot tell.
    """
    try:
        import mc_llm_runtime

        return bool(mc_llm_runtime.runtime.running())
    except Exception:
        return False


def _phase_for(status: str) -> str:
    """Which phase a status line from the session means we have reached.

    The session says what it is doing in prose, for a status area; this maps
    that back onto a phase. Only one transition matters -- the writer's own
    status is emitted immediately before the request goes out, so it is the
    start of prompt evaluation -- and everything else is still waiting.
    """
    from prompt_master.krea import enhancer

    text = str(status or "").casefold()
    if enhancer.label(0).casefold() in text or "writing the krea prompt" in text:
        return mc_progress.PHASE_KREA_READ
    return mc_progress.PHASE_KREA_WAIT


def _directed(recipe) -> str:
    """One line saying what the Director chose, for the status area.

    The count and the seed, not the brief. The brief is shown in full in the
    diagnostic view where somebody has asked to see it; a status line that
    printed nine sentences of art direction would push everything else off the
    panel every time somebody pressed Generate.
    """
    count = len(getattr(recipe, "items", ()))
    verb = "Replayed" if getattr(recipe, "replayed", False) else "Directed"
    if not count:
        return f"No creative direction at Creativity {recipe.creativity}."
    return (f"{verb} {count} {'axis' if count == 1 else 'axes'} · "
            f"Creative seed {recipe.creative_seed}")


creative = Creative()
"""The one Creative Mode session.

A module singleton for the reason ``mc_llm_runtime.runtime`` and the broker are:
one browser, one GPU, one llama-server, one Forge. It holds the last roll so the
diagnostics drawer has something to show, and nothing else -- there is no gap
between a roll and the generation that uses it for anything else to live in.
"""
