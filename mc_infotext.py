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
