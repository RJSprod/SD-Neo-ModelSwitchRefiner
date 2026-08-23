"""Infotext writing and paste-field registration for the Model Chain extension.

Every control is written into ``p.extra_generation_params`` under a namespaced
key and mapped back onto its UI component on paste (section 7). Keys are
omitted entirely when the extension is disabled, so ordinary generations keep a
clean infotext.

Escaping is handled by the host: ``create_infotext`` runs every value through
``infotext_utils.quote()``, which JSON-quotes anything containing a comma, colon
or newline, and ``parse_generation_parameters`` runs the matching ``unquote()``.
That is what keeps a resolved prompt -- including ``<lora:name:weight>`` tags,
which contain colons -- intact across the round trip.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #

ENABLED = "Model Chain Enabled"
TARGET = "Model Chain Target"
PROMPT_MODE = "Model Chain Prompt Mode"
PROMPT = "Model Chain Prompt"
NEGATIVE = "Model Chain Negative"
STYLES = "Model Chain Styles"
SEED_MODE = "Model Chain Seed Mode"
SEED_OFFSET = "Model Chain Seed Offset"
SEED_FIXED = "Model Chain Seed"
"""Not in the section 7.2 list, but the batch-wide fixed seed is unrecoverable
without it, so "Fixed" mode would not survive the round trip."""
CFG = "Model Chain CFG"
STEPS = "Model Chain Steps"
SAMPLER = "Model Chain Sampler"
SCHEDULER = "Model Chain Scheduler"
DENOISE = "Model Chain Denoise"
SIZE_MULTIPLIER = "Model Chain Size Multiplier"
STAGE1_SIZE = "Model Chain Stage1 Size"
EDIT_MODE = "Model Chain Edit Mode"
"""Auto | Enable | Disable -- Stage 2 reference/edit conditioning."""

REFERENCE_MODE = "Model Chain Reference Mode"
"""Disabled | Pass Through ImageStitch | Decoupled -- supplemental Stage 2 references."""

REFERENCE_COUNT = "Model Chain References"
"""How many supplemental references the Stage 2 pass was given.

Diagnostic only, and deliberately so. Native ImageStitch records no reference
pixels in infotext and neither does this: the count says a set was in play and
how large it was, which is what makes a result reproducible *by hand*, but the
images themselves have to be re-supplied exactly as ImageStitch requires today.

It is settled before the infotexts are written, which is the last moment the
Stage 1 model is still loaded and infotext can be built at all. In the rare case
where the loaded Stage 2 model then turns out not to accept the set, the images
are dropped and a notice says so on the result -- so this reads as the set Stage
2 was handed, which is the same number in every ordinary generation.
"""

REFERENCE_MAX_DIM = "Model Chain Reference Max Side"
"""Longest side the Decoupled references were resized to before encoding.

Recorded for Decoupled only. In Pass Through the sizing belongs to ImageStitch's
own "Maximum Side Length" control, and recording Model Chain's unused slider
there would describe something that did not happen.
"""

REFERENCE_DEFAULT_MODE = "Disabled"
"""Omitted from infotext, so a chain that never touches this feature keeps the
infotext it had before the feature existed."""

REFERENCE_DECOUPLED_MODE = "Decoupled"
"""The one mode whose reference sizing Model Chain owns."""

# --------------------------------------------------------------------------- #
# Krea Creative Mode
# --------------------------------------------------------------------------- #

CREATIVE_MODE = "Krea Creative Mode"
CREATIVE_CREATIVITY = "Krea Creativity"
CREATIVE_SEED = "Krea Creative Seed"
CREATIVE_LLM_SEED = "Krea LLM Seed"
CREATIVE_RECIPE = "Krea Creative Recipe"
CREATIVE_SOURCE = "Krea Source Prompt"
CREATIVE_LIBRARY = "Krea Creativity Library"
CREATIVE_LORAS = "Krea Pinned LoRAs"
"""What images made before the literal syntax recorded, and nothing writes now.

The Pinned LoRAs control is gone and ``[[<lora:name:weight>]]`` in the prompt is
what replaced it. The key stays readable because the images that carry it still
exist and "what the pasted image records" should be able to say what tags that
picture used -- read and shown, never applied, never written again.
"""

CREATIVE_AXES = "Krea Creative Axes"
CREATIVE_EXCLUDED = "Krea Creative Excluded"
CREATIVE_ANTI = "Krea Anti Repetition"
CREATIVE_WRITER = "Krea Writer Model"
"""What a Creative Mode generation records, and what it deliberately does not.

There are two different questions here and conflating them is what made the old
behaviour wrong.

**How do I make this picture again?** Answered by the image's own ``Prompt:``
line and nothing else. Creative Mode assigns the expanded prompt to ``p.prompt``
before Forge records infotext, so the recorded Prompt *is* the paragraph the
image model was given -- which means an ordinary paste reproduces the image by
restoring it and switching Creative Mode **off**. Leaving Creative Mode on would
hand that paragraph back to the writer as a fresh idea and expand it a second
time, and the result would be a picture of the prompt of the picture. That is
what :func:`build_creative_paste_fields` exists to prevent.

**How do I get back to the workflow that made it?** Answered by the keys above.
The source phrase is the short thing the user typed, which nothing else in the
file preserves. The Creative seed reproduces the local roll, the recipe ids say
what that roll chose without needing it re-rolled, the library version says which
vocabulary those ids came from, the axes and exclusions say what configuration
allowed them, and the writer model says which language model turned the brief
into English. Restoring all of that is a separate, explicit action on the panel,
because it overwrites the prompt box -- and a paste button that silently replaced
what somebody was iterating on would be the more destructive of the two failures.

The expanded Krea prompt is still absent from this list, and still for the reason
it always was: it is the ``Prompt:`` line. Recording it a second time under a key
of ours would add a few hundred bytes to every PNG to repeat what the file
already says, and would create a second copy a later paste could disagree with.
"""

CREATIVE_KEYS = (CREATIVE_MODE, CREATIVE_CREATIVITY, CREATIVE_SEED, CREATIVE_LLM_SEED,
                 CREATIVE_RECIPE, CREATIVE_SOURCE, CREATIVE_LIBRARY, CREATIVE_LORAS,
                 CREATIVE_AXES, CREATIVE_EXCLUDED, CREATIVE_ANTI, CREATIVE_WRITER)
"""Every key a Creative generation may write, for forwarding and for tests."""

SPATIAL_MODE = "Krea Spatial Mode"
SPATIAL_VERSION = "Krea Spatial Version"
SPATIAL_LAYOUT = "Krea Spatial Layout"
SPATIAL_COMPOSE_MODE = "Krea Spatial Compose Mode"
SPATIAL_COMPOSER_MODEL = "Krea Spatial Composer Model"
SPATIAL_COMPOSER_SEED = "Krea Spatial Composer Seed"
SPATIAL_COMPOSER_VERSION = "Krea Spatial Composer Instruction"
SPATIAL_PROMPT_VERSION = "Krea Spatial Prompt Version"
SPATIAL_ENHANCED_SCENE = "Krea Enhanced Scene"
"""What a Smart image used to call the scene it handed the Composer.

Read, never written. The name was true while Spatial could only run behind
Creative Mode and stopped being true the day it could run on its own -- with the
writer off, that scene is the user's own sentence and nothing enhanced it.
:data:`SPATIAL_INPUT_SCENE` is the neutral name new images use; this one stays so
old images still say what they said.
"""

SPATIAL_INPUT_SCENE = "Krea Spatial Input Scene"
"""The scene supplied to the Spatial Composer before reconciliation.

Creative on: the Creative Writer's paragraph. Creative off: the prompt exactly
as typed. One key for one thing, whichever produced it.
"""

SPATIAL_SOURCE = "Krea Spatial Source"
"""Where the scene came from, when it did not come from the Creative Writer.

Written as ``prompt`` on a Spatial-only generation and absent otherwise, so the
question "was this composed around a written scene or around what I typed" has
an answer in the file rather than being inferred from which other keys are
missing.
"""

SPATIAL_SCENE = "Krea Spatial Scene"
"""The scene the Composer returned, after reconciliation. Smart merges only."""

_SPATIAL_NAMESPACE = """What a Spatial BBOX generation records, in its own namespace.

The same two questions the Creative keys answer, asked about the composition.

**How do I make this picture again?** Still the image's own ``Prompt:`` line and
still nothing else -- it is the finished structured prompt, aspect ratio, scene,
background, elements and all, assigned to ``p.prompt`` before Forge wrote the
infotext. An ordinary paste therefore reproduces a spatial image exactly, and
does it by restoring that prompt and switching *both* Creative Mode and Spatial
Layout off. Leaving either on would rebuild the prompt from the layout around a
scene that is already a finished structured prompt, and the result would be a
BBOX prompt with a BBOX prompt inside it.

**How do I get back to the canvas?** These keys. The layout is the whole
editable state, normalised: version, canvas, compose mode, and every region with
its box, its words, its type, its framing, its angle and its z-order. That is
several hundred bytes on a spatial image and nothing at all on any other, which
is the trade §8.2 asks for -- and the layout is the one thing in the file that
cannot be reconstructed from the prompt, because the prompt carries the derived
hints rather than the selections they came from.

The two scene keys are diagnostic and are written only in Smart mode, where the
scene handed to the Composer really is unrecoverable afterwards. In Direct mode
that scene *is* the ``high_level_description``, and recording it twice would
repeat the file to itself.

A Spatial-only generation writes these and no Creative keys at all, which is
what lets its paste switch Spatial Layout off without a Creative key being
present to switch anything.
"""

LITERAL_VERSION = "Krea Literal Syntax Version"
LITERAL_COUNT = "Krea Literal Command Count"
LITERAL_KEYS = (LITERAL_VERSION, LITERAL_COUNT)
"""What a generation records about the ``[[literal commands]]`` in its prompt.

Two numbers, and deliberately not the payloads. The payloads are in the image's
own ``Prompt:`` line, which is what the model was given, and in
:data:`CREATIVE_SOURCE` with their brackets still on, which is what the user
typed. A third copy would be bytes repeating the file and a copy a later paste
could disagree with.

The count is what makes a silent failure visible after the fact: an image whose
prompt looks as though a command went missing either records a count that says
otherwise, or records nothing at all and never had one. The version is here for
the same reason every other version key in this module is -- the syntax is small
and will grow, and an image made under version 1 should be read as version 1.

Written by Creative, Spatial and plain generations alike, in their own
namespace, because a literal command is a property of the prompt rather than of
either feature.
"""

SPATIAL_KEYS = (SPATIAL_MODE, SPATIAL_VERSION, SPATIAL_LAYOUT, SPATIAL_COMPOSE_MODE,
                SPATIAL_COMPOSER_MODEL, SPATIAL_COMPOSER_SEED,
                SPATIAL_COMPOSER_VERSION, SPATIAL_PROMPT_VERSION,
                SPATIAL_ENHANCED_SCENE, SPATIAL_INPUT_SCENE, SPATIAL_SOURCE,
                SPATIAL_SCENE)
"""Every key a Spatial generation may write, for forwarding and for tests."""

KLEIN_SPATIAL_MODE = "Klein Spatial Mode"
KLEIN_SPATIAL_RESOLVED = "Klein Spatial Resolved Mode"
KLEIN_SPATIAL_VERSION = "Klein Spatial Version"
KLEIN_SPATIAL_LAYOUT = "Klein Spatial Layout"
KLEIN_SPATIAL_SOURCE = "Klein Spatial Source"
KLEIN_SPATIAL_SOURCE_COUNT = "Klein Spatial Source Count"
KLEIN_SPATIAL_BACKEND = "Klein Spatial Regional Backend"
KLEIN_SPATIAL_BACKEND_VERSION = "Klein Spatial Regional Backend Version"
KLEIN_SPATIAL_ATTACHED = "Klein Spatial Regions Attached"
"""How many regions actually conditioned the image, out of how many were drawn.

An observation and not an intention, and recorded because the two came apart:
for several runs the boxes were drawn, compiled and encoded and then reached
nothing, and the images looked exactly like images where they had. "3 of 3" and
"0 of 3" are the difference between a spatial generation and an ordinary one
wearing its metadata.
"""

KLEIN_SPATIAL_COMPOSE_MODE = "Klein Spatial Compose Mode"
KLEIN_SPATIAL_COMPOSER_SEED = "Klein Spatial Composer Seed"
KLEIN_SPATIAL_COMPOSER_VERSION = "Klein Spatial Composer Instruction"
"""What the Smart Compose pass did to the global prompt, if it ran.

Its own three keys rather than Krea's, for the same reason the rest of this
namespace is separate: the same Composer instruction runs for both backends and
what it *produces* lands in two different places -- a field of a structured
document for Krea, the whole prompt the model reads for Klein. An image that
recorded ``Krea Spatial Composer Seed`` and was made by Klein would be readable
and wrong.

The composed prompt itself is not recorded, and does not need to be: unlike
Krea's, it *is* the image's own ``Prompt:`` line. The seed and the instruction
version are what make "why did the same layout compose differently today"
answerable.
"""

KLEIN_SPATIAL_KEYS = (KLEIN_SPATIAL_MODE, KLEIN_SPATIAL_RESOLVED,
                      KLEIN_SPATIAL_VERSION, KLEIN_SPATIAL_LAYOUT,
                      KLEIN_SPATIAL_SOURCE, KLEIN_SPATIAL_SOURCE_COUNT,
                      KLEIN_SPATIAL_BACKEND, KLEIN_SPATIAL_BACKEND_VERSION,
                      KLEIN_SPATIAL_COMPOSE_MODE, KLEIN_SPATIAL_COMPOSER_SEED,
                      KLEIN_SPATIAL_COMPOSER_VERSION, KLEIN_SPATIAL_ATTACHED)

_KLEIN_SPATIAL_NAMESPACE = """What a FLUX.2 Klein spatial generation records.

A separate namespace from the Krea keys above, and §37 is unusually firm about
why: the two backends consume the same canvas and mean different things by it,
so an old ``Krea Spatial Layout`` record must never be read as a Klein
regional-conditioning job. Separate prefixes make that a property of the file
format rather than of a reader remembering to check.

**How do I make this picture again?** Not from the ``Prompt:`` line alone, and
this is the one real difference from Krea (§39). Krea's structured prompt carries
its boxes as text, so an ordinary paste reproduces a Krea spatial image exactly.
Klein's regional conditioning never touches ``p.prompt`` -- the recorded prompt
is the user's own global prompt -- so an image pasted back without its layout
reproduces the scene and not the composition. :data:`KLEIN_SPATIAL_LAYOUT` is
therefore essential metadata rather than a convenience, and it is why a paste
does not silently switch Spatial Layout *off* the way a Krea paste must: there is
no double-composition to prevent here.

It does not silently switch it *on* either. §39 asks for both halves: an ordinary
paste restores the ordinary prompt and settings, and "Restore Spatial setup" is
the explicit action that puts the canvas, the requested mode and the source
expectation back.

**Which mechanism made it?** :data:`KLEIN_SPATIAL_BACKEND` and its version. Two
images made from the same layout by two different regional-conditioning
mechanisms are two different experiments, and a file that recorded only "regional
conditioning happened" could not tell them apart afterwards.

:data:`KLEIN_SPATIAL_SOURCE_COUNT` is an expectation and never a payload: it says
the image was made with N ImageStitch references so that a restore into an empty
gallery can say so out loud. The reference pixels themselves are not in the file,
which is the same policy Stage 2 references already follow.
"""

MODULE_PREFIX = "Model Chain Module "
"""Numbered VAE / text encoder keys: "Model Chain Module 1", "... 2", and so on.

Mirrors how the host records its own ``Module N`` and ``Hires Module N``
entries, including the ``Use same choices`` / ``Built-in`` sentinels in slot 1.
"""

INHERIT_MODULES = "Use same choices"
BUILTIN_MODULES = "Built-in"

PROMPT_MODES = ("Inherit", "Append", "Replace")
SEED_MODES = ("Inherit", "Offset", "Fixed")

INHERIT = "Same as Stage 1"
"""Sentinel for the sampler and schedule-type dropdowns."""

# Kept as an alias: the sampler control shipped with this name.
INHERIT_SAMPLER = INHERIT


def build_params(
    *,
    target: str,
    prompt_mode: str,
    prompt,
    negative,
    styles: list[str] | None,
    seed_mode: str,
    seed_offset: int,
    fixed_seed: int,
    cfg: float,
    steps: int,
    sampler: str,
    denoise: float,
    size_multiplier: float,
    stage1_size: str,
    scheduler: str = INHERIT,
    edit_mode: str = "Auto",
    modules=None,
    reference_mode: str = REFERENCE_DEFAULT_MODE,
    reference_count: int = 0,
    reference_max_dim: int | None = None,
) -> dict:
    """Build the ``extra_generation_params`` entries for an enabled chain.

    ``prompt`` and ``negative`` accept either a string or a per-image list;
    ``create_infotext`` indexes list values by image index, which is how a
    prompt that varies across the batch is recorded accurately per image.
    """
    params = {
        ENABLED: "True",
        TARGET: target,
        PROMPT_MODE: prompt_mode,
        PROMPT: prompt,
        NEGATIVE: negative,
        SEED_MODE: seed_mode,
        CFG: round(float(cfg), 4),
        STEPS: int(steps),
        DENOISE: round(float(denoise), 4),
        SIZE_MULTIPLIER: round(float(size_multiplier), 4),
        STAGE1_SIZE: stage1_size,
    }

    if styles:
        params[STYLES] = ", ".join(styles)

    # Omitted when zero, per section 7.2.
    if seed_mode == "Offset" and int(seed_offset) != 0:
        params[SEED_OFFSET] = int(seed_offset)

    if seed_mode == "Fixed":
        params[SEED_FIXED] = int(fixed_seed)

    if sampler and sampler != INHERIT:
        params[SAMPLER] = sampler

    if scheduler and scheduler != INHERIT:
        params[SCHEDULER] = scheduler

    # Omitted at its default so a chain that does not touch edit mode keeps the
    # same infotext it had before the feature existed.
    if edit_mode and edit_mode != "Auto":
        params[EDIT_MODE] = edit_mode

    # Same reasoning as edit mode: the default is omitted so an image made
    # without this feature carries no trace of it.
    if reference_mode and reference_mode != REFERENCE_DEFAULT_MODE:
        params[REFERENCE_MODE] = reference_mode
        if int(reference_count or 0) > 0:
            params[REFERENCE_COUNT] = int(reference_count)
        if reference_mode == REFERENCE_DECOUPLED_MODE and reference_max_dim is not None:
            params[REFERENCE_MAX_DIM] = int(reference_max_dim)

    params.update(build_module_params(modules))

    return {k: v for k, v in params.items() if v not in (None, "")}


def build_module_params(modules) -> dict:
    """Record the Stage 2 VAE / text encoder selection.

    Follows the host's own convention for ``Hires Module N``: the sentinel goes
    in slot 1 when the selection is inherited or empty, otherwise each module is
    recorded by its extension-less basename.
    """
    import os

    if modules is None:
        return {}
    if isinstance(modules, str):
        modules = [modules]

    if INHERIT_MODULES in modules:
        # Inheriting is the default; recording it would add noise to every
        # infotext for a setting the user never touched.
        return {}
    if not modules:
        return {f"{MODULE_PREFIX}1": BUILTIN_MODULES}

    return {
        f"{MODULE_PREFIX}{i + 1}": os.path.splitext(os.path.basename(str(m)))[0]
        for i, m in enumerate(modules)
    }


def parse_modules(params: dict) -> list[str] | None:
    """Collect ``Model Chain Module N`` keys back into a selection.

    Returns None when the infotext records no selection, which restores the
    inherit default. Names are matched against the host's live module list the
    same way the host resolves its own ``Module N`` keys, so a module that has
    since been removed is dropped rather than breaking the paste.
    """
    import os

    numbered = []
    for key, value in params.items():
        if not key.startswith(MODULE_PREFIX):
            continue
        suffix = key[len(MODULE_PREFIX) :].strip()
        if suffix.isdigit():
            numbered.append((int(suffix), value))

    if not numbered:
        return None

    numbered.sort()
    values = [value for _, value in numbered]

    if values[0] == INHERIT_MODULES:
        return [INHERIT_MODULES]
    if values[0] == BUILTIN_MODULES:
        return []

    try:
        from modules_forge.main_entry import module_list

        known = {os.path.splitext(name)[0]: name for name in module_list}
    except Exception:
        return None

    resolved, missing = [], []
    for value in values:
        name = known.get(str(value))
        (resolved if name else missing).append(name or value)

    if missing:
        logger.warning(
            "Model Chain: VAE/text encoder modules from this infotext are not "
            "installed: %s",
            ", ".join(str(m) for m in missing),
        )

    return resolved


def parse_styles(value) -> list[str]:
    """Split a recorded ``Model Chain Styles`` value back into names.

    Style names containing a comma cannot survive this round trip. That is
    acceptable because the *resolved* prompt is what reproduces the image
    (section 7.3); the names are recorded for human readability and are pruned
    against the live store on paste.
    """
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


# --------------------------------------------------------------------------- #
# Paste fields
# --------------------------------------------------------------------------- #


def _lookup_checkpoint(title):
    """Resolve a recorded checkpoint name against the installed checkpoints.

    Returns None when the model is gone, which leaves the dropdown untouched;
    the caller surfaces a non-fatal notice naming it (section 7.3).
    """
    if not title:
        return None
    try:
        from modules import sd_models

        info = sd_models.get_closet_checkpoint_match(title)
    except Exception:
        return None
    return None if info is None else info.title


def build_paste_fields(components: dict) -> list:
    """Map every infotext key back onto its UI component.

    Unknown or missing keys fall back to the control's current value because
    ``_parse_info`` skips a field whose key is absent -- infotext from an older
    version of the extension therefore never throws (section 7.3).
    """
    import gradio as gr
    from modules.infotext_utils import PasteField

    def styles_from(params):
        names = parse_styles(params.get(STYLES))
        if not names:
            return None

        from mc_styles import prune_selection

        kept, missing = prune_selection(names)
        if missing:
            logger.warning(
                "Model Chain: styles named in this infotext no longer exist: %s "
                "(the resolved Stage 2 prompt was restored regardless)",
                ", ".join(missing),
            )
        return gr.update(value=kept)

    def target_from(params):
        recorded = params.get(TARGET)
        if not recorded:
            return None
        resolved = _lookup_checkpoint(recorded)
        if resolved is None:
            logger.warning(
                "Model Chain: checkpoint %r from this infotext is not installed; "
                "the Stage 2 model was left unset",
                recorded,
            )
            return None
        return resolved

    def modules_from(params):
        selection = parse_modules(params)
        if selection is None:
            return gr.update(value=[INHERIT_MODULES])
        return gr.update(value=selection)

    def edit_mode_from(params):
        # Absent means the image predates the feature, or was made in Auto.
        # Restoring Auto is right in both cases; returning None would leave
        # whatever the control happened to be showing.
        return params.get(EDIT_MODE, "Auto")

    def reference_mode_from(params):
        mode = params.get(REFERENCE_MODE, REFERENCE_DEFAULT_MODE)
        if mode != REFERENCE_DEFAULT_MODE:
            # Said on paste rather than left for the failed generation: the
            # images are not in the infotext and never were, exactly as native
            # ImageStitch behaves, so the mode restores and the reference set
            # has to be rebuilt by hand.
            logger.info(
                'Model Chain: restored Stage 2 reference mode "%s"%s. Reference images are '
                "not stored in PNG info — re-add them before generating, or Stage 2 will "
                "run without supplemental references.",
                mode,
                f" ({params[REFERENCE_COUNT]} image(s) were used)" if REFERENCE_COUNT in params else "",
            )
        return mode

    fields = [
        PasteField(components["enabled"], lambda d: ENABLED in d, api="model_chain_enabled"),
        PasteField(components["target"], target_from, api="model_chain_target"),
        PasteField(components["prompt_mode"], PROMPT_MODE, api="model_chain_prompt_mode"),
        PasteField(components["prompt"], PROMPT, api="model_chain_prompt"),
        PasteField(components["negative"], NEGATIVE, api="model_chain_negative"),
        PasteField(components["styles"], styles_from, api="model_chain_styles"),
        PasteField(components["seed_mode"], SEED_MODE, api="model_chain_seed_mode"),
        PasteField(components["seed_offset"], SEED_OFFSET, api="model_chain_seed_offset"),
        PasteField(components["fixed_seed"], SEED_FIXED, api="model_chain_fixed_seed"),
        PasteField(components["cfg"], CFG, api="model_chain_cfg"),
        PasteField(components["steps"], STEPS, api="model_chain_steps"),
        PasteField(components["sampler"], SAMPLER, api="model_chain_sampler"),
        PasteField(components["scheduler"], SCHEDULER, api="model_chain_scheduler"),
        PasteField(components["denoise"], DENOISE, api="model_chain_denoise"),
        PasteField(components["size_multiplier"], SIZE_MULTIPLIER, api="model_chain_size_multiplier"),
        PasteField(components["edit_mode"], edit_mode_from, api="model_chain_edit_mode"),
        PasteField(components["modules"], modules_from, api="model_chain_modules"),
        PasteField(
            components["reference_mode"], reference_mode_from, api="model_chain_reference_mode"
        ),
        PasteField(
            components["reference_max_dim"],
            REFERENCE_MAX_DIM,
            api="model_chain_reference_max_dim",
        ),
    ]

    return fields


# --------------------------------------------------------------------------- #
# Krea Creative Mode: writing the configuration down, and reading it back
# --------------------------------------------------------------------------- #


def creative_axes(modes, fixed) -> str:
    """The axis configuration as one line: ``medium=fixed:oil_impasto, mood=vary``.

    Natural axes are left out, which is what keeps this short on the ordinary
    configuration: a fresh install directs nothing and writes nothing, and a user
    who has configured two axes gets two entries rather than ten. Absence is not
    ambiguous here -- everything unlisted is Natural, which is what Natural means.

    Ids, never labels. A label is display text that a package update may rewrite;
    an id is stable by the package's own contract, which is the only reason a
    recorded configuration is worth restoring at all.
    """
    from prompt_master.krea import director

    modes = modes or {}
    fixed = fixed or {}
    entries = []
    for key, mode in modes.items():
        mode = str(mode).casefold()
        if mode == director.FIXED:
            pinned = str(fixed.get(key) or "")
            entries.append(f"{key}=fixed:{pinned}" if pinned else f"{key}=fixed")
        elif mode == director.VARY:
            entries.append(f"{key}=vary")
    return ", ".join(entries)


def parse_creative_axes(value) -> tuple[dict, dict]:
    """``"medium=fixed:oil_impasto, mood=vary"`` back into ``(modes, fixed ids)``.

    Only the axes the line names. The caller fills the rest with Natural, because
    that is what their absence meant when the line was written -- and because an
    image made before an axis existed must not be able to say anything about it.
    """
    from prompt_master.krea import director

    modes, fixed = {}, {}
    for entry in _entries(value):
        key, _, described = entry.partition("=")
        key, described = key.strip(), described.strip()
        if not key or not described:
            continue
        mode, _, pinned = described.partition(":")
        mode = mode.strip().casefold()
        if mode not in director.MODES:
            continue
        modes[key] = mode
        if mode == director.FIXED and pinned.strip():
            fixed[key] = pinned.strip()
    return modes, fixed


def creative_exclusions(excluded) -> str:
    """Excluded ids as one line: ``lighting=harsh_noon|golden_hour, texture=gloss``.

    Pipes inside an axis, commas between them, because the host's own quoting
    already understands a comma and this way one axis's exclusions stay one
    token even when a value is read by eye.
    """
    entries = []
    for key, values in (excluded or {}).items():
        chosen = [str(value) for value in (values or ()) if str(value)]
        if chosen:
            entries.append(f"{key}={'|'.join(chosen)}")
    return ", ".join(entries)


def parse_creative_exclusions(value) -> dict:
    """One recorded exclusions line back into ``{axis: [id, ...]}``."""
    excluded = {}
    for entry in _entries(value):
        key, _, listed = entry.partition("=")
        key = key.strip()
        chosen = [part.strip() for part in listed.split("|") if part.strip()]
        if key and chosen:
            excluded[key] = chosen
    return excluded


def _entries(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(entry).strip() for entry in value if str(entry).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def creative_setup(params: dict):
    """One infotext's Creative fields as a :class:`mc_creative_krea.Setup`.

    Reading, and only reading. Nothing here applies anything: the panel's restore
    button does that, when somebody presses it, and this is what it reads.
    """
    import mc_creative_krea

    modes, fixed = parse_creative_axes(params.get(CREATIVE_AXES))
    excluded = parse_creative_exclusions(params.get(CREATIVE_EXCLUDED))
    return mc_creative_krea.Setup(
        source=str(params.get(CREATIVE_SOURCE) or ""),
        creativity=_number(params.get(CREATIVE_CREATIVITY)),
        seed=_number(params.get(CREATIVE_SEED)),
        llm_seed=_number(params.get(CREATIVE_LLM_SEED)),
        recipe=str(params.get(CREATIVE_RECIPE) or ""),
        library_version=str(params.get(CREATIVE_LIBRARY) or ""),
        axis_modes=modes,
        fixed_values=fixed,
        excluded_values=excluded,
        anti_repetition=_flag(params.get(CREATIVE_ANTI)),
        loras=str(params.get(CREATIVE_LORAS) or ""),
        writer=str(params.get(CREATIVE_WRITER) or ""),
        spatial_layout=str(params.get(SPATIAL_LAYOUT) or ""),
        spatial_compose_mode=str(params.get(SPATIAL_COMPOSE_MODE) or ""),
        spatial_version=_number(params.get(SPATIAL_VERSION)),
        klein_layout=str(params.get(KLEIN_SPATIAL_LAYOUT) or ""),
        klein_mode=str(params.get(KLEIN_SPATIAL_MODE) or ""),
        klein_resolved_mode=str(params.get(KLEIN_SPATIAL_RESOLVED) or ""),
        klein_source=str(params.get(KLEIN_SPATIAL_SOURCE) or ""),
        klein_source_count=_number(params.get(KLEIN_SPATIAL_SOURCE_COUNT)),
        klein_backend=str(params.get(KLEIN_SPATIAL_BACKEND) or ""),
        klein_compose_mode=str(params.get(KLEIN_SPATIAL_COMPOSE_MODE) or ""),
    )


def _number(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _flag(value):
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip().casefold() in ("true", "1", "yes", "on")


def build_creative_paste_fields(components: dict, notice=None, view=None,
                                spatial_view=None) -> list:
    """Map a pasted Creative infotext onto the txt2img Creative controls.

    Exactly one control is *changed* by an ordinary paste, and it is switched
    off. That is the whole of the reproduction fix:

    * the recorded ``Prompt:`` line is the expanded prompt the image model was
      given, and the host restores it as it restores any prompt;
    * Creative Mode is switched off, so nothing expands it a second time;
    * the seed, checkpoint, sampler, size and everything else are the host's and
      are untouched by this.

    The rest of the Creative record is *captured* rather than applied -- stashed
    on :data:`mc_creative_krea.pasted` for the panel's explicit "Restore Creative
    setup" button, which is the action that is allowed to overwrite the prompt
    box, because somebody pressed it.

    A paste with no Creative keys in it changes nothing here. Returning ``None``
    from a paste field is how the host is told to leave a control alone, and an
    ordinary image should not be able to switch a feature off any more than on.
    """
    from modules.infotext_utils import PasteField

    import mc_creative_krea

    def creative_off(params):
        if CREATIVE_MODE not in params:
            return None
        logger.info("Model Chain: this image was made with Krea Creative Mode. Its "
                    "final expanded prompt has been restored and Creative Mode was "
                    "switched off, so the prompt is not expanded a second time.")
        return False

    def spatial_off(params):
        """The same answer, for the same reason, one layer out.

        A spatial image's recorded Prompt is the finished structured prompt --
        aspect ratio, scene, background and every element. Restoring it with
        Spatial Layout still on would put that whole document into
        ``high_level_description`` and build a second BBOX prompt around it,
        which reproduces nothing and is not even obviously wrong to look at.

        Two switches rather than one because they are two features. Creative
        Mode off with Spatial on would compose boxes around a prompt nobody
        expanded; a spatial image has to switch off both, and an image carrying
        neither key must be able to switch off neither.
        """
        if SPATIAL_MODE not in params:
            return None
        logger.info("Model Chain: this image was made with Krea Spatial Layout. Its "
                    "final structured prompt has been restored and Spatial Layout was "
                    "switched off, so the layout is not composed onto it a second "
                    "time.")
        return False

    def captured(params):
        """Stash the Creative record, and say on the panel what just happened.

        Piggy-backed on a paste field because a paste field is the only callback
        the host offers on a paste, and the capture has to happen on every paste
        rather than on the ones somebody remembers to press a button after.
        """
        setup = creative_setup(params)
        mc_creative_krea.pasted.remember(setup)
        # A paste is a new starting point; an armed replay from an earlier one
        # is not part of it and would silently direct the next generation.
        mc_creative_krea.replay.clear()
        if not setup.present:
            return None
        said = ("Creative image restored using its final expanded prompt. Creative Mode "
                "was disabled to prevent re-expansion. Its source prompt and settings "
                "are under Creative Controls → Continue from a pasted image.")
        return notice(said) if notice else said

    def pasted_view(params):
        # The surface hands its own view function in rather than being imported
        # by name: this module is imported by the host under a name it chooses,
        # and a helper module reaching back into a script by string would be a
        # dependency nothing declares.
        if CREATIVE_MODE not in params or view is None:
            return None
        return view()

    def spatial_pasted_view(params):
        if spatial_view is None:
            return None
        if SPATIAL_MODE not in params and KLEIN_SPATIAL_MODE not in params:
            return None
        return spatial_view()

    def spatial_said(params):
        """The Spatial section's own line about what a paste just did.

        Its own, because a Spatial-only image has no Creative record and the
        Creative line would not be written for it -- so without this a paste
        that switched Spatial Layout off would do it silently, and the next
        press would look like the feature had stopped working.
        """
        if KLEIN_SPATIAL_MODE in params:
            # A Klein spatial image is the opposite case, and §39 asks for the
            # opposite behaviour. Its recorded Prompt is the user's own global
            # prompt -- regional conditioning never rewrote it -- so there is no
            # second composition to prevent and nothing to switch off. What the
            # paste must *not* do is switch Spatial Layout on, because a canvas
            # arriving with a picture is not somebody asking to compose with it.
            said = ("This image was made with Spatial Layout on FLUX.2 Klein. Its "
                    f"prompt and settings have been restored; the "
                    f"{_klein_restore_hint(params)} is under Spatial options → "
                    "Continue from a pasted image, and Restore Spatial setup puts "
                    "it back.")
            return notice(said) if notice else said
        if SPATIAL_MODE not in params:
            return None
        said = ("Spatial image restored using its final structured prompt. Spatial "
                "Layout was disabled so the layout is not composed onto it a second "
                "time. The canvas it was drawn on is under Spatial options → "
                "Continue from a pasted image.")
        return notice(said) if notice else said

    fields = [PasteField(components["enabled"], creative_off, api="krea_creative_enabled")]
    if "spatial_enabled" in components:
        fields.append(PasteField(components["spatial_enabled"], spatial_off,
                                 api="krea_spatial_enabled"))
    if "status" in components:
        fields.append(PasteField(components["status"], captured,
                                 api="krea_creative_status"))
    if "pasted" in components and view is not None:
        fields.append(PasteField(components["pasted"], pasted_view,
                                 api="krea_creative_pasted"))
    if "spatial_pasted" in components and spatial_view is not None:
        fields.append(PasteField(components["spatial_pasted"], spatial_pasted_view,
                                 api="krea_spatial_pasted"))
    if "spatial_status" in components:
        fields.append(PasteField(components["spatial_status"], spatial_said,
                                 api="krea_spatial_status"))
    return fields


def _klein_restore_hint(params: dict) -> str:
    """How to describe what a pasted Klein spatial image left to restore.

    Names the references when the image had them, because that is the half a
    restore cannot supply: the layout is in the file and the reference pixels
    deliberately are not, so a restore into an empty gallery has to say so rather
    than present an image-required mode that will refuse at the next press.
    """
    count = _number(params.get(KLEIN_SPATIAL_SOURCE_COUNT)) or 0
    if count > 0:
        return (f"canvas it was drawn on (made with {count} ImageStitch reference "
                f"image{'' if count == 1 else 's'}, which are not stored in the file)")
    return "canvas it was drawn on"


def creative_paste_field_names() -> list[str]:
    """Keys the "Send to txt2img" buttons must forward for the above to work.

    All of them, including the ones only the restore button reads: the buttons
    forward by exact name, so a key that is not listed here simply does not
    arrive, and "restore the setup" would find half a record.
    """
    return (list(CREATIVE_KEYS) + list(SPATIAL_KEYS) + list(KLEIN_SPATIAL_KEYS)
            + list(LITERAL_KEYS))


def paste_field_names() -> list[str]:
    """Keys this extension wants forwarded by the "Send to txt2img" buttons."""
    return [
        ENABLED,
        TARGET,
        PROMPT_MODE,
        PROMPT,
        NEGATIVE,
        STYLES,
        SEED_MODE,
        SEED_OFFSET,
        SEED_FIXED,
        CFG,
        STEPS,
        SAMPLER,
        SCHEDULER,
        DENOISE,
        SIZE_MULTIPLIER,
        STAGE1_SIZE,
        EDIT_MODE,
        REFERENCE_MODE,
        REFERENCE_COUNT,
        REFERENCE_MAX_DIM,
        # A generous fixed span: the host forwards keys by exact name, and a
        # chain realistically selects a VAE plus one or two text encoders.
        *(f"{MODULE_PREFIX}{i}" for i in range(1, 9)),
    ]
