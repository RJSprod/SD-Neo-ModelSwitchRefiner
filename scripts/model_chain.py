"""Model Chain -- cross-architecture two-stage generation for Forge Neo.

Stage 1 completes an ordinary txt2img generation on the loaded checkpoint.
Stage 2 re-encodes the finished *pixels* and runs an img2img refinement pass on
a second checkpoint. Because the handoff is in pixel space rather than latent
space, each model uses its own VAE and text encoder, so any A -> B pairing the
WebUI can load will work -- SDXL -> Flux.2-Klein included.

This is emphatically not a latent-space handoff, and it does not modify or
replace the built-in Refiner, which switches the diffusion model mid-sampling
and is therefore restricted to models sharing a latent space.
"""

from __future__ import annotations

import logging
import os
import time

import gradio as gr

import mc_arch
import mc_infotext
import mc_lora
import mc_memory
import mc_presets
import mc_references
import mc_styles
from modules import errors, images, processing, scripts, shared
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from modules.shared import opts, state
from modules.ui_common import refresh_symbol
from modules.ui_components import InputAccordion, ToolButton

logger = mc_memory.logger
"""Shared with the helper modules; mc_memory attaches the console handler."""

STAGE1_SUBFOLDER = "model-chain-stage1"

_NO_MODEL = "None"

_REFINED_MARKER = "_model_chain_refined"
"""Set on a Processed once Stage 2 has run over it."""

_TRANSITION_DESCRIPTIONS = {
    "warm": "warm swap from the RAM cache, no disk read",
    "cold": "cold load from disk",
    "unchanged": "already loaded",
}

DEFAULT_DENOISE = 1.0
"""Full-strength Stage 2 pass by default.

Lower it to keep more of the Stage 1 composition. It also matches what
reference/edit conditioning expects, so enabling edit mode needs no
adjustment."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

SETTINGS_SECTION = ("model_chain", "Model Chain")
"""Identifier and title of the Settings page section.

The first element must not be None. ``ui_settings`` skips any option whose
section id is None -- that is the host's own idiom for settings it stores but
deliberately never draws, which is how ``sd_checkpoint_hash`` and
``disabled_extensions`` stay out of the Settings page. Registered with None here
these options exist and can be read and saved, but no control is ever built for
them, so there is nothing to tick.
"""

shared.options_templates.update(
    shared.options_section(
        SETTINGS_SECTION,
        {
            "model_chain_save_stage1": shared.OptionInfo(
                False,
                "Save Stage 1 intermediate images to disk",
            ).info(
                f'written to a "{STAGE1_SUBFOLDER}" subfolder of the output directory; '
                "they never appear in the gallery"
            ),
            "model_chain_ram_budget_gb": shared.OptionInfo(
                0.0,
                "Max system RAM for model cache (GB)",
                gr.Number,
            ).info(
                "0 uses a default of 60% of detected system RAM. The live free-RAM "
                "check is the real guard, so this is a ceiling rather than a reservation"
            ),
            # The two below describe the machine rather than the image, which is
            # why they live here and not in the accordion: an accordion control
            # would also travel in presets and in every infotext, where a VRAM
            # strategy says nothing useful about the hardware that reads it.
            mc_memory.OPT_PRELOAD: shared.OptionInfo(
                mc_memory.PRELOAD_DEFAULT,
                "Preload Stage 1 after Stage 2 finishes (experimental)",
            ).info(
                "moves Stage 1's weights back into VRAM in the background while you look "
                "at the result, so the next Generate starts sampling immediately. Off by "
                "default: it is the only part of this extension that touches models off "
                "the generation thread, and on some setups — on-the-fly LoRA patching in "
                "particular — that has broken the following generation outright. It saves "
                "no work, it only moves it earlier, so leave it off unless you have "
                "confirmed it is safe on your machine"
            ),
            mc_memory.OPT_PIN_ENCODERS: shared.OptionInfo(
                True,
                "Keep Stage 1's text encoder and VAE in VRAM during Stage 2",
            ).info(
                "so switching back only has to move the UNet. Applied only when Stage 2 "
                "still fits alongside them; otherwise it is skipped automatically and logged"
            ),
            mc_memory.OPT_WARM_STAGE_2: shared.OptionInfo(
                True,
                "Use leftover VRAM to keep Stage 2 warm between generations",
            ).info(
                "after Stage 1 is back and the reserve below is honoured, anything still "
                "spare is spent on Stage 2's components so the next refine starts sooner. "
                "Never at Stage 1's expense, and the first thing released under pressure. "
                "Requires the preload above: filling VRAM is only safe on that thread, so "
                "with the preload off this does nothing"
            ),
            mc_memory.OPT_VRAM_RESERVE: shared.OptionInfo(
                0.0,
                "Minimum VRAM reserve (GB)",
                gr.Number,
            ).info(
                "0 sizes the reserve automatically from the pass and from the activation "
                "peaks actually observed this session. Set a number to put a floor under "
                "that estimate — Model Chain never warms anything into the reserve, and "
                "never undercuts VRAM Forge has already set aside for itself"
            ),
            mc_memory.OPT_PRESERVE_LORA: shared.OptionInfo(
                True,
                "Reuse a prepared LoRA state when a cached model comes back",
            ).info(
                "a warm swap keeps the model object, so an unchanged LoRA does not have to "
                "be reapplied. Turn this off to make every restored model rebuild its LoRA "
                "state from scratch — slower, and worth trying if a LoRA misbehaves after "
                "a switch"
            ),
        },
    )
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _model_choices() -> tuple[list[str], list[str]]:
    """Checkpoint and VAE/text-encoder choices, rescanned from disk.

    Module choices carry the same "Use same choices" sentinel the host's own
    Hires VAE/TE dropdown uses.
    """
    try:
        from modules_forge.main_entry import refresh_models

        ckpts, modules = refresh_models()
        return [_NO_MODEL] + list(ckpts), [mc_memory.INHERIT_MODULES] + list(modules)
    except Exception:
        logger.warning("Model Chain: failed to list checkpoints", exc_info=True)
        return [_NO_MODEL], [mc_memory.INHERIT_MODULES]


def _sampler_choices() -> list[str]:
    from modules import sd_samplers

    return [mc_infotext.INHERIT] + list(sd_samplers.visible_sampler_names())


def _scheduler_choices() -> list[str]:
    from modules import sd_schedulers

    # Labels, not names -- that is what the host's own schedule-type dropdowns
    # offer, and schedulers_map accepts either.
    return [mc_infotext.INHERIT] + [x.label for x in sd_schedulers.schedulers]


def _resolution_note(
    width: int,
    height: int,
    multiplier: float,
    target: str,
    hires: bool = False,
    hr_scale: float = 2.0,
    hr_resize_x: int = 0,
    hr_resize_y: int = 0,
) -> str:
    """Live description of the Stage 2 output size (section 6.5).

    The main width/height sliders describe the *first pass*. With hires fix on,
    Stage 2 receives the upscaled image, so the readout has to follow the
    upscale or it describes an image that never reaches Stage 2.
    """
    try:
        width, height = int(width), int(height)
        multiplier = float(multiplier)
    except (TypeError, ValueError):
        return ""

    if width <= 0 or height <= 0:
        return ""

    source_note = ""
    if hires:
        try:
            width, height = mc_arch.hires_target_size(
                width, height, float(hr_scale or 2.0), int(hr_resize_x or 0), int(hr_resize_y or 0)
            )
            source_note = " after Hires. fix"
        except (TypeError, ValueError, ZeroDivisionError):
            return ""

    arch = mc_arch.detect_from_checkpoint_name(target)
    out_w, out_h = mc_arch.scaled_size(width, height, multiplier, arch.alignment)

    suffix = f"aligned to {arch.alignment}px"
    if arch is not mc_arch.UNKNOWN:
        suffix += f" for {arch.label}"
    else:
        suffix += " (architecture unknown)"

    note = (
        f"Stage 2 output: **{out_w} x {out_h}** — from {width} x {height}"
        f"{source_note}, {suffix}."
    )

    delta = mc_arch.aspect_ratio_delta((width, height), (out_w, out_h))
    if delta > 0.01:
        note += f" Aspect ratio shifts by {delta * 100:.1f}% to reach the grid."
    return note


def _edit_notice(target: str, mode: str) -> str:
    """Explain what edit mode will do for the selected Stage 2 model."""
    if not target or target == _NO_MODEL:
        return ""

    arch = mc_arch.detect_from_checkpoint_name(target)

    if not arch.supports_edit:
        if mode == mc_arch.EDIT_AUTO:
            return ""
        label = arch.label if arch is not mc_arch.UNKNOWN else "This model"
        if arch.supports_references:
            # Flux.1 Kontext and Qwen-Image-Edit reference, but there is no
            # setting behind it: the loader decides from the checkpoint. Saying
            # this control is simply "ignored" would be wrong now that it also
            # vetoes Model Chain's own Stage 2 references.
            return (
                f"⚠️ {label} has no edit toggle — whether it uses reference conditioning is "
                'decided by the checkpoint variant, not by this control. "Disable" still stops '
                "Model Chain from supplying Stage 2 reference images."
            )
        return f"⚠️ {label} has no edit/reference mode — this setting will be ignored."

    if not mc_arch.edit_is_active(arch, mode):
        return (
            f"Stage 2 runs as plain img2img. {arch.label} can instead take the Stage 1 "
            f'image as an *edit reference* — set this to "Enable".'
        )

    note = (
        f"**{arch.label} edit mode.** The Stage 1 image is passed as an edit "
        f"reference (vision conditioning + reference latents) rather than as an "
        f"img2img starting point, so denoise should be at or near "
        f"{arch.edit_denoise:g}."
    )
    if arch.edit_needs_lora:
        note += (
            f" {arch.label} needs its Edit LoRA for this to work — add it to the "
            "Stage 2 prompt with `<lora:name:weight>` in Append or Replace mode."
        )
    if mode == mc_arch.EDIT_AUTO:
        note += " (Currently on via the global Settings toggle.)"
    return note


def _reference_notice(mode: str, target: str, stitch_gallery=None, decoupled_gallery=None) -> str:
    """Status and compatibility text for the Stage 2 reference control.

    Two separate things are said, because they fail independently: how many
    supplemental references the selected mode currently has, and whether the
    selected Stage 2 model can consume any at all.
    """
    if not mc_references.is_active(mode):
        return ""

    lines = []

    if mode == mc_references.PASS_THROUGH:
        if stitch_gallery is None:
            # No gallery to read. Either ImageStitch is not installed at all, or
            # it is but its gallery never reached after_component -- an id
            # rename in a later Forge. Only the *live* count is lost in the
            # second case; the generation reads the same gallery out of the
            # script arguments regardless.
            if mc_references.stitch_is_installed():
                lines.append(
                    "Stage 2 will use whatever reference images ImageStitch holds when you generate."
                )
            else:
                lines.append(
                    "⚠️ ImageStitch is not installed, so there are no references to pass through. "
                    'Use "Decoupled" to supply Stage 2 references from Model Chain instead.'
                )
        else:
            count = len(mc_references.extract_images(stitch_gallery))
            if count:
                lines.append(
                    f"Using **{count}** ImageStitch reference image(s) for each Stage 2 image, "
                    "in the order shown there."
                )
            else:
                lines.append(
                    "⚠️ ImageStitch has no reference images, so Stage 2 will run without "
                    "supplemental references."
                )
    else:
        count = len(mc_references.extract_images(decoupled_gallery))
        if count:
            lines.append(
                f"Using **{count}** Model Chain reference image(s) for each Stage 2 image. "
                "Stage 1's own ImageStitch references are not included."
            )
        else:
            lines.append(
                "⚠️ No Stage 2 reference images have been added, so Stage 2 will run without "
                "supplemental references."
            )

    compatibility = _reference_compatibility_notice(target)
    if compatibility:
        lines.append(compatibility)

    return "\n\n".join(lines)


def _reference_compatibility_notice(target: str) -> str:
    """Whether the selected Stage 2 model can take supplemental references."""
    if not target or target == _NO_MODEL:
        return ""

    arch = mc_arch.detect_from_checkpoint_name(target)

    if arch is mc_arch.UNKNOWN:
        return (
            "⚠️ The Stage 2 architecture could not be detected, so reference support is "
            "unknown. It is checked again once the model loads, and the references are "
            "dropped with a notice if it has no reference path."
        )

    if not arch.supports_references:
        return (
            f"⚠️ **{arch.label} has no reference-conditioning path**, so supplemental "
            "reference images will be ignored. Stage 2 still receives its own Stage 1 image "
            "as usual."
        )

    if not arch.references_are_validated:
        note = (
            f"⚠️ **{arch.label} reference routing is experimental** — Forge exposes a "
            "reference path for it, but this pairing is not one Model Chain has validated. "
            "Flux.2 Klein and Krea 2 are."
        )
        if arch.reference_flag in ("kontext", "edit"):
            note += (
                f" Only the reference-capable {arch.label} variants have that path at all; "
                "an ordinary checkpoint will have its references dropped with a notice."
            )
        return note

    return ""


def _architecture_notice(target: str) -> str:
    """Warn about the Stage 1 -> Stage 2 architecture boundary (section 6.7)."""
    if not target or target == _NO_MODEL:
        return ""

    stage2 = mc_arch.detect_from_checkpoint_name(target)
    stage1 = mc_arch.detect_loaded()

    if stage1 is mc_arch.UNKNOWN or stage2 is mc_arch.UNKNOWN or stage1.key == stage2.key:
        return ""

    return (
        f"**Stage 1 is {stage1.label}, Stage 2 is {stage2.label}.** "
        "LoRAs, embeddings and ControlNet units active in the main prompt are "
        "Stage-1-architecture-specific and will **not** carry into Stage 2. "
        "To use a LoRA during refinement, add it to the Stage 2 prompt using "
        "`<lora:name:weight>` syntax."
    )


# --------------------------------------------------------------------------- #
# Script
# --------------------------------------------------------------------------- #


class ScriptModelChain(scripts.Script):
    """Two-stage pixel-space chain, registered as an alwayson script."""

    def __init__(self):
        super().__init__()
        self._width_component = None
        self._height_component = None
        self._hires_component = None
        self._hr_scale_component = None
        self._hr_resize_x_component = None
        self._hr_resize_y_component = None
        # Guards Stage 2's own process_images() call from re-entering this
        # script. Stage 2 runs with p.scripts unset, so this is belt and
        # braces -- but the cost of getting it wrong is an infinite chain.
        self._in_stage_2 = False
        self._armed = False
        # Whether the host would have written this generation's images to disk,
        # captured before Stage 1 saving is suppressed.
        self._save_final_images = False
        # Extra-network tags last dropped from an inherited prompt, so the
        # console says so once per generation rather than once per image.
        self._dropped_networks: list[str] = []
        # The loaded model's geometry state as Stage 1 was about to sample.
        # Captured there because that is the only moment it describes Stage 1;
        # by the time a wrong size is noticed, Model B is loaded.
        self._stage1_geometry = ""
        # The two measurements either side of the noise build, which is where a
        # requested size stops being a number and becomes a tensor.
        self._stage1_requested = ()
        self._stage1_latent = ""
        # ImageStitch's own Reference Image(s) gallery, captured for the live
        # count in Pass Through mode. Read-only, and optional: without it the
        # panel says less, and the generation itself is unaffected because it
        # reads the same gallery out of the script arguments.
        self._stitch_gallery_component = None
        # Set by ui() when the reference status has to wait for that gallery,
        # and consumed the moment it arrives.
        self._wire_reference_notice = None
        # This generation's supplemental reference set, captured in process()
        # while the Stage 1 script arguments are still live.
        self._references = ()

    def title(self):
        return "Model Chain"

    def show(self, is_img2img):
        # txt2img only: Stage 1 is a txt2img generation by definition.
        return scripts.AlwaysVisible if not is_img2img else None

    # -- UI --------------------------------------------------------------- #

    # Main-tab controls the size readout depends on. The hires ones matter
    # because Stage 2 refines the *upscaled* image, not the first pass.
    _TRACKED_COMPONENTS = {
        "txt2img_width": "_width_component",
        "txt2img_height": "_height_component",
        "txt2img_hr-checkbox": "_hires_component",
        "txt2img_hr_scale": "_hr_scale_component",
        "txt2img_hr_resize_x": "_hr_resize_x_component",
        "txt2img_hr_resize_y": "_hr_resize_y_component",
    }

    def after_component(self, component, **kwargs):
        """Capture main-tab controls that drive the live readouts."""
        elem_id = kwargs.get("elem_id") or ""

        attribute = self._TRACKED_COMPONENTS.get(elem_id)
        if attribute is not None:
            setattr(self, attribute, component)

        # Matched on the tail rather than the whole id: Script.elem_id prefixes
        # the tab name for a script shown on both tabs, and ImageStitch is one.
        if elem_id.endswith(mc_references.STITCH_GALLERY_SUFFIX) and self._stitch_gallery_component is None:
            self._stitch_gallery_component = component
            self._wire_deferred_reference_notice(component)

    def _wire_deferred_reference_notice(self, gallery) -> None:
        """Finish the reference status now ImageStitch's gallery exists.

        A one-shot: the wiring is dropped once used, so a second gallery with a
        matching id -- or a rebuilt UI -- cannot register the same handlers
        twice and leave two dependencies racing for the same output.
        """
        wire, self._wire_reference_notice = self._wire_reference_notice, None
        if wire is None:
            return
        try:
            wire(gallery)
        except Exception:
            errors.report("Model Chain: failed to wire the reference status", exc_info=True)

    def ui(self, is_img2img):
        checkpoints, module_choices = _model_choices()
        samplers = _sampler_choices()
        schedulers = _scheduler_choices()
        styles = mc_styles.available_styles()

        with InputAccordion(False, label="Model Chain", elem_id=self.elem_id("enable")) as enabled:
            gr.Markdown(
                "Finishes Stage 1 on the loaded checkpoint, then re-encodes the "
                "result and refines it with a second checkpoint. The handoff is in "
                "**pixel space**, so the two models may use different architectures, "
                "VAEs and text encoders."
            )

            # -- presets --------------------------------------------------- #
            with gr.Row():
                preset = gr.Dropdown(
                    value=mc_presets.NONE,
                    label="Preset",
                    choices=mc_presets.choices(),
                    elem_id=self.elem_id("preset"),
                    info="selecting a preset applies it immediately",
                )
                preset_refresh = ToolButton(
                    value=refresh_symbol,
                    elem_id=self.elem_id("preset_refresh"),
                    tooltip="Presets: refresh",
                )
            with gr.Row():
                preset_name = gr.Textbox(
                    label="Preset name",
                    placeholder="name to save the current Stage 2 settings under",
                    elem_id=self.elem_id("preset_name"),
                    scale=3,
                )
                preset_save = gr.Button("Save", elem_id=self.elem_id("preset_save"), scale=1)
                preset_delete = gr.Button("Delete", elem_id=self.elem_id("preset_delete"), scale=1)
            preset_status = gr.Markdown("", elem_id=self.elem_id("preset_status"))

            with gr.Row():
                target = gr.Dropdown(
                    value=_NO_MODEL,
                    label="Stage 2 checkpoint",
                    choices=checkpoints,
                    elem_id=self.elem_id("target"),
                )
                target_refresh = ToolButton(
                    value=refresh_symbol,
                    elem_id=self.elem_id("target_refresh"),
                    tooltip="Stage 2 checkpoint and modules: refresh",
                )

            with gr.Row():
                modules = gr.Dropdown(
                    value=[mc_memory.INHERIT_MODULES],
                    label="Stage 2 VAE / Text Encoder",
                    choices=module_choices,
                    multiselect=True,
                    elem_id=self.elem_id("modules"),
                    info=(
                        f'"{mc_memory.INHERIT_MODULES}" keeps Stage 1\'s selection; '
                        "clear it entirely to use the checkpoint's built-in modules"
                    ),
                )

            architecture_notice = gr.Markdown("", elem_id=self.elem_id("arch_notice"))
            residency_status = gr.Markdown("", elem_id=self.elem_id("residency"))

            # -- prompt ---------------------------------------------------- #
            with gr.Group():
                prompt_mode = gr.Radio(
                    choices=list(mc_infotext.PROMPT_MODES),
                    value="Inherit",
                    label="Stage 2 prompt",
                    elem_id=self.elem_id("prompt_mode"),
                )
                gr.Markdown(
                    "Flux-family models respond to natural-language phrasing rather "
                    "than comma-separated tags, so a Stage 2 prompt often needs "
                    "different wording than Stage 1. `<lora:name:weight>` tags here "
                    "are applied against the Stage 2 model.",
                    elem_id=self.elem_id("prompt_hint"),
                )

                prompt = gr.Textbox(
                    label="Stage 2 positive",
                    lines=2,
                    placeholder="Added to (Append) or replacing (Replace) the Stage 1 prompt",
                    elem_id=self.elem_id("prompt"),
                    visible=False,
                )
                negative = gr.Textbox(
                    label="Stage 2 negative",
                    lines=2,
                    elem_id=self.elem_id("negative"),
                    visible=False,
                )

                with gr.Row():
                    style_selection = gr.Dropdown(
                        label="Stage 2 styles",
                        choices=styles,
                        value=[],
                        multiselect=True,
                        interactive=False,
                        elem_id=self.elem_id("styles"),
                        tooltip="Applied to the Stage 2 prompt; ignored in Inherit mode",
                    )
                    style_refresh = ToolButton(
                        value=refresh_symbol,
                        elem_id=self.elem_id("styles_refresh"),
                        tooltip="Stage 2 styles: refresh",
                    )

            # -- sampling -------------------------------------------------- #
            with gr.Group():
                with gr.Row():
                    denoise = gr.Slider(
                        label="Denoise strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=DEFAULT_DENOISE,
                        elem_id=self.elem_id("denoise"),
                        info="how much Stage 2 may alter the Stage 1 image",
                    )
                    size_multiplier = gr.Slider(
                        label="Output size multiplier",
                        minimum=1.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        elem_id=self.elem_id("size_multiplier"),
                    )

                size_note = gr.Markdown("", elem_id=self.elem_id("size_note"))

                with gr.Row():
                    steps = gr.Slider(
                        label="Stage 2 steps",
                        minimum=1,
                        maximum=150,
                        step=1,
                        value=20,
                        elem_id=self.elem_id("steps"),
                    )
                    cfg = gr.Slider(
                        label="Stage 2 CFG scale",
                        minimum=1.0,
                        maximum=30.0,
                        step=0.5,
                        value=7.0,
                        elem_id=self.elem_id("cfg"),
                    )

                with gr.Row():
                    sampler = gr.Dropdown(
                        label="Stage 2 sampling method",
                        choices=samplers,
                        value=mc_infotext.INHERIT,
                        elem_id=self.elem_id("sampler"),
                    )
                    scheduler = gr.Dropdown(
                        label="Stage 2 schedule type",
                        choices=schedulers,
                        value=mc_infotext.INHERIT,
                        elem_id=self.elem_id("scheduler"),
                    )

            # -- edit / reference conditioning ----------------------------- #
            with gr.Group():
                edit_mode = gr.Radio(
                    choices=list(mc_arch.EDIT_MODES),
                    value=mc_arch.EDIT_AUTO,
                    label="Stage 2 edit mode",
                    elem_id=self.elem_id("edit_mode"),
                    info="Auto follows the global Settings toggle for the model",
                )
                edit_notice = gr.Markdown("", elem_id=self.elem_id("edit_notice"))

            # -- supplemental reference images ----------------------------- #
            with gr.Group():
                reference_mode = gr.Radio(
                    choices=list(mc_references.MODES),
                    value=mc_references.DISABLED,
                    label="Stage 2 reference images",
                    elem_id=self.elem_id("reference_mode"),
                    info=(
                        "supplemental references, shared by every Stage 2 image and in "
                        "addition to each image's own Stage 1 handoff"
                    ),
                )
                reference_notice = gr.Markdown("", elem_id=self.elem_id("reference_notice"))

                with gr.Group(visible=False) as reference_panel:
                    reference_images = gr.Gallery(
                        value=None,
                        type="pil",
                        interactive=True,
                        show_label=False,
                        container=False,
                        show_download_button=False,
                        show_share_button=False,
                        label="Stage 2 Reference Image(s)",
                        min_width=512,
                        height=384,
                        columns=3,
                        rows=1,
                        allow_preview=False,
                        object_fit="contain",
                        elem_id=self.elem_id("reference_images"),
                    )

                    reference_selected = gr.State(-1)

                    with gr.Row():
                        reference_upload = gr.Image(
                            height=225,
                            width=225,
                            sources="upload",
                            type="pil",
                            label="Image to Upload",
                            show_download_button=False,
                            show_share_button=False,
                            elem_id=self.elem_id("reference_upload"),
                        )
                        with gr.Column():
                            reference_append = gr.Button(
                                "Append Pasted Image", elem_id=self.elem_id("reference_append")
                            )
                            reference_replace = gr.Button(
                                "Replace Selected Image", elem_id=self.elem_id("reference_replace")
                            )
                            reference_delete = gr.Button(
                                "Delete Selected Image",
                                variant="stop",
                                elem_id=self.elem_id("reference_delete"),
                            )
                            reference_clear = gr.Button(
                                "Clear All References",
                                variant="stop",
                                elem_id=self.elem_id("reference_clear"),
                            )

                reference_max_dim = gr.Slider(
                    minimum=0,
                    maximum=2048,
                    value=mc_references.DEFAULT_MAX_DIM,
                    step=256,
                    label="Reference maximum side length",
                    visible=False,
                    elem_id=self.elem_id("reference_max_dim"),
                    info=(
                        "reduces VRAM during encoding; 0 for no limit. Pass Through uses "
                        "ImageStitch's own setting instead"
                    ),
                )

            # -- seed ------------------------------------------------------ #
            with gr.Group():
                seed_mode = gr.Radio(
                    choices=list(mc_infotext.SEED_MODES),
                    value="Inherit",
                    label="Stage 2 seed",
                    elem_id=self.elem_id("seed_mode"),
                    info="Inherit reuses each image's own Stage 1 seed",
                )
                with gr.Row():
                    seed_offset = gr.Number(
                        label="Seed offset",
                        value=0,
                        precision=0,
                        visible=False,
                        elem_id=self.elem_id("seed_offset"),
                    )
                    fixed_seed = gr.Number(
                        label="Fixed seed",
                        value=-1,
                        precision=0,
                        visible=False,
                        elem_id=self.elem_id("fixed_seed"),
                    )

        # -- wiring -------------------------------------------------------- #

        def on_prompt_mode(mode):
            uses_text = mode in ("Append", "Replace")
            return (
                gr.update(visible=uses_text),
                gr.update(visible=uses_text),
                # Inherit ignores styles entirely, so make that visible rather
                # than silently dropping the selection (section 5.2).
                gr.update(interactive=uses_text),
            )

        prompt_mode.change(
            fn=on_prompt_mode,
            inputs=[prompt_mode],
            outputs=[prompt, negative, style_selection],
            show_progress=False,
        )

        def on_seed_mode(mode):
            return gr.update(visible=mode == "Offset"), gr.update(visible=mode == "Fixed")

        seed_mode.change(
            fn=on_seed_mode,
            inputs=[seed_mode],
            outputs=[seed_offset, fixed_seed],
            show_progress=False,
        )

        def on_model_refresh(selected_modules):
            """Rescan checkpoints and modules, keeping selections that survive."""
            checkpoint_choices, refreshed_modules = _model_choices()
            kept = [m for m in (selected_modules or []) if m in refreshed_modules]
            if not kept:
                kept = [mc_memory.INHERIT_MODULES]
            return gr.update(choices=checkpoint_choices), gr.update(
                choices=refreshed_modules, value=kept
            )

        target_refresh.click(
            fn=on_model_refresh,
            inputs=[modules],
            outputs=[target, modules],
            show_progress=False,
        )

        def on_style_refresh(selected):
            names = mc_styles.reload_styles()
            kept, missing = mc_styles.prune_selection(selected)
            if missing:
                logger.info("Model Chain: dropped styles removed from the library: %s", ", ".join(missing))
            return gr.update(choices=names, value=kept)

        style_refresh.click(
            fn=on_style_refresh,
            inputs=[style_selection],
            outputs=[style_selection],
            show_progress=False,
        )

        def on_target_change(target_name, multiplier, mode, selected_modules, width, height, *hires):
            """Refresh the architecture notice, residency status and size readout."""
            # The module selection changes the footprint and the cache key, so
            # the residency prediction has to account for it.
            plan = mc_memory.plan(target_name, selected_modules)
            status = plan.message
            if plan.is_warning:
                status = f"⚠️ {status}"

            arch = mc_arch.detect_from_checkpoint_name(target_name)
            cfg_update = gr.skip()
            if arch is not mc_arch.UNKNOWN:
                # Architecture-aware CFG default (section 6.4): defaulting a
                # distilled Flux model to 7.0 produces poor output.
                cfg_update = gr.update(value=arch.cfg)

            return (
                _architecture_notice(target_name),
                status,
                _resolution_note(width, height, multiplier, target_name, *hires),
                cfg_update,
                _edit_notice(target_name, mode),
            )

        target_outputs = [architecture_notice, residency_status, size_note, cfg, edit_notice]

        # Controls the size readout reads. The hires ones are optional: a
        # heavily customised UI may not expose them, and the readout degrades
        # to the first-pass size rather than showing nothing.
        hires_components = [
            self._hires_component,
            self._hr_scale_component,
            self._hr_resize_x_component,
            self._hr_resize_y_component,
        ]
        have_hires = all(component is not None for component in hires_components)
        have_size = self._width_component is not None and self._height_component is not None

        def size_inputs():
            base = [self._width_component, self._height_component, size_multiplier, target]
            return base + hires_components if have_hires else base

        if have_size:
            target_inputs = [target, size_multiplier, edit_mode, modules,
                             self._width_component, self._height_component]
            if have_hires:
                target_inputs += hires_components

            def on_target(*values):
                # Positional to stay in step with target_inputs, which grows
                # when the hires controls are available.
                name, mult, mode, mods, width, height = values[:6]
                return on_target_change(name, mult, mode, mods, width, height, *values[6:])

            for trigger in (target, modules):
                trigger.change(
                    fn=on_target,
                    inputs=target_inputs,
                    outputs=target_outputs,
                    show_progress=False,
                )

            watched = [size_multiplier, self._width_component, self._height_component]
            if have_hires:
                watched += hires_components
            for component in watched:
                component.change(
                    fn=_resolution_note,
                    inputs=size_inputs(),
                    outputs=[size_note],
                    show_progress=False,
                )
        else:
            # The main sliders were not found (a heavily customised UI); the
            # size readout is simply omitted rather than showing wrong numbers.
            logger.info("Model Chain: txt2img size sliders unavailable, live size readout disabled")
            for trigger in (target, modules):
                trigger.change(
                    fn=lambda name, mult, mode, mods: on_target_change(name, mult, mode, mods, 0, 0),
                    inputs=[target, size_multiplier, edit_mode, modules],
                    outputs=target_outputs,
                    show_progress=False,
                )

        def on_edit_mode(target_name, mode, current_denoise):
            """Raise denoise to what edit conditioning expects, if it was lowered.

            Edit mode hands the model a reference rather than a starting image,
            so the reference carries the content and a lowered denoise is the
            wrong control. The slider is moved rather than silently overridden
            at generate time, so the value stays visible.

            Turning edit mode *off* deliberately leaves the slider alone: the
            default is already a full pass, so there is nothing to restore, and
            a value the user chose themselves should survive the toggle.
            """
            arch = mc_arch.detect_from_checkpoint_name(target_name)
            denoise_update = gr.skip()

            if mc_arch.edit_is_active(arch, mode) and current_denoise < arch.edit_denoise:
                denoise_update = gr.update(value=arch.edit_denoise)

            return _edit_notice(target_name, mode), denoise_update

        edit_mode.change(
            fn=on_edit_mode,
            inputs=[target, edit_mode, denoise],
            outputs=[edit_notice, denoise],
            show_progress=False,
        )

        # -- reference wiring ---------------------------------------------- #
        #
        # The live status counts whichever gallery the selected mode reads, so
        # ImageStitch's own gallery is an input to it. It is only ever read: the
        # user's Stage 1 selection is never written to, reordered or cleared
        # from here.

        def on_reference_change(mode, target_name, decoupled_gallery, stitch_gallery=None):
            decoupled = mode == mc_references.DECOUPLED
            return (
                gr.update(visible=decoupled),
                gr.update(visible=decoupled),
                _reference_notice(mode, target_name, stitch_gallery, decoupled_gallery),
            )

        reference_outputs = [reference_panel, reference_max_dim, reference_notice]

        def wire_reference_notice(stitch_gallery=None):
            """Connect the status once it is settled what it can read.

            ImageStitch sorts below Model Chain, so its gallery does not exist
            yet while this panel is being built -- a Gradio event captures its
            input list at registration, so wiring now would permanently bind a
            status that cannot see it. When ImageStitch is installed the wiring
            therefore waits for ``after_component`` to hand the gallery over;
            when it is not, there is nothing to wait for and this runs at once.

            ``target`` is a trigger as well as an input: the compatibility half
            of the notice is about the Stage 2 architecture, so it has to follow
            the checkpoint.
            """
            inputs = [reference_mode, target, reference_images]
            if stitch_gallery is not None:
                inputs.append(stitch_gallery)

            for trigger in inputs:
                trigger.change(
                    fn=on_reference_change,
                    inputs=inputs,
                    outputs=reference_outputs,
                    show_progress=False,
                )

        if mc_references.stitch_is_installed():
            self._wire_reference_notice = wire_reference_notice
        else:
            wire_reference_notice()

        def on_reference_select(event: gr.SelectData):
            return event.index

        reference_images.select(
            fn=on_reference_select,
            outputs=[reference_selected],
            queue=False,
            show_progress=False,
        )

        # Append / replace / delete / clear, kept deliberately close to
        # ImageStitch's own gallery controls so the two panels behave alike.
        # Every one of them rebuilds an ordered list: reference order carries
        # meaning, so nothing here is allowed to reshuffle it.
        def on_reference_append(gallery, image):
            if image is None:
                return [gr.skip(), gr.skip()]
            gallery = list(gallery or [])
            gallery.append((image, None))
            return [gr.update(value=gallery), gr.update(value=None)]

        def on_reference_replace(index, gallery, image):
            if image is None or not gallery or index < 0 or index >= len(gallery):
                return [-1, gr.skip(), gr.skip()]
            gallery = list(gallery)
            gallery[index] = (image, None)
            return [-1, gr.update(value=gallery), gr.update(value=None)]

        def on_reference_delete(index, gallery):
            if not gallery or index < 0 or index >= len(gallery):
                return [-1, gr.skip()]
            gallery = list(gallery)
            gallery.pop(index)
            return [-1, gr.update(value=gallery)]

        reference_append.click(
            fn=on_reference_append,
            inputs=[reference_images, reference_upload],
            outputs=[reference_images, reference_upload],
            queue=False,
            show_progress=False,
        )
        reference_replace.click(
            fn=on_reference_replace,
            inputs=[reference_selected, reference_images, reference_upload],
            outputs=[reference_selected, reference_images, reference_upload],
            queue=False,
            show_progress=False,
        )
        reference_delete.click(
            fn=on_reference_delete,
            inputs=[reference_selected, reference_images],
            outputs=[reference_selected, reference_images],
            queue=False,
            show_progress=False,
        )
        reference_clear.click(
            fn=lambda: [-1, gr.update(value=[])],
            outputs=[reference_selected, reference_images],
            queue=False,
            show_progress=False,
        )

        components = {
            "enabled": enabled,
            "target": target,
            "modules": modules,
            "prompt_mode": prompt_mode,
            "prompt": prompt,
            "negative": negative,
            "styles": style_selection,
            "seed_mode": seed_mode,
            "seed_offset": seed_offset,
            "fixed_seed": fixed_seed,
            "cfg": cfg,
            "steps": steps,
            "sampler": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
            "size_multiplier": size_multiplier,
            "edit_mode": edit_mode,
            "reference_mode": reference_mode,
            "reference_images": reference_images,
            "reference_max_dim": reference_max_dim,
        }

        # -- preset wiring ------------------------------------------------- #
        #
        # Ordered to match mc_presets.FIELDS so a preset's values map onto the
        # controls positionally, the same contract ui()'s return value has with
        # the processing hooks.
        preset_controls = [components[name] for name in mc_presets.FIELDS]
        preset_defaults = {name: components[name].value for name in mc_presets.FIELDS}

        def on_preset_selected(name):
            if not name or name == mc_presets.NONE:
                return ["", *([gr.skip()] * len(preset_controls))]

            values = mc_presets.get(name)
            if values is None:
                return [
                    f'⚠️ Preset "{name}" no longer exists — refresh the list.',
                    *([gr.skip()] * len(preset_controls)),
                ]

            resolved = mc_presets.apply_defaults(values, preset_defaults)
            logger.info("Model Chain: applied preset %r", name)
            return [
                f'Applied preset "{name}".',
                *[gr.update(value=resolved[field]) for field in mc_presets.FIELDS],
            ]

        preset.change(
            fn=on_preset_selected,
            inputs=[preset],
            outputs=[preset_status, *preset_controls],
            show_progress=False,
        )

        def on_preset_save(name, *values):
            try:
                saved = mc_presets.save(name, dict(zip(mc_presets.FIELDS, values)))
            except mc_presets.PresetError as exc:
                return f"⚠️ {exc}", gr.skip()
            return (
                f'Saved preset "{name.strip()}".',
                gr.update(choices=[mc_presets.NONE] + saved, value=name.strip()),
            )

        preset_save.click(
            fn=on_preset_save,
            inputs=[preset_name, *preset_controls],
            outputs=[preset_status, preset],
            show_progress=False,
        )

        def on_preset_delete(name):
            try:
                remaining = mc_presets.delete(name)
            except mc_presets.PresetError as exc:
                return f"⚠️ {exc}", gr.skip()
            return (
                f'Deleted preset "{name}".',
                gr.update(choices=[mc_presets.NONE] + remaining, value=mc_presets.NONE),
            )

        preset_delete.click(
            fn=on_preset_delete,
            inputs=[preset],
            outputs=[preset_status, preset],
            show_progress=False,
        )

        def on_preset_refresh(current):
            available = mc_presets.names()
            keep = current if current in available else mc_presets.NONE
            return gr.update(choices=[mc_presets.NONE] + available, value=keep)

        preset_refresh.click(
            fn=on_preset_refresh,
            inputs=[preset],
            outputs=[preset],
            show_progress=False,
        )

        try:
            self.infotext_fields = mc_infotext.build_paste_fields(components)
            self.paste_field_names = mc_infotext.paste_field_names()
        except Exception:
            errors.report("Model Chain: failed to register paste fields", exc_info=True)

        return [
            enabled,
            target,
            modules,
            prompt_mode,
            prompt,
            negative,
            style_selection,
            seed_mode,
            seed_offset,
            fixed_seed,
            cfg,
            steps,
            sampler,
            scheduler,
            denoise,
            size_multiplier,
            edit_mode,
            reference_mode,
            reference_images,
            reference_max_dim,
        ]

    # -- prompt assembly --------------------------------------------------- #

    def _resolve_prompts(self, stage1_positive, stage1_negative, mode, extra_positive, extra_negative, styles):
        """Assemble one Stage 2 prompt pair.

        Order is prompt assembly -> style expansion -> (host) extra-networks
        parsing, per section 5.4. Tags the user wrote into the Stage 2 boxes are
        never touched here: the assembled string is handed to the normal
        processing pipeline, which parses and strips them exactly as it does for
        the main prompt box.

        The Stage 1 half is the exception, and is stripped of extra-network tags
        before it is used. ``all_prompts`` holds the prompt as typed, tags
        included -- the host strips them from a copy on the way to the text
        encoder, which is why they survive into infotext -- so inheriting it
        verbatim would apply Stage 1's LoRAs against Model B. That is at its
        worst in exactly the case this extension exists for, where the two
        models are different architectures and the LoRA lands on the wrong
        tensors or fails outright.
        """
        if mode == "Replace":
            positive, negative = extra_positive, extra_negative
        else:
            inherited_positive, dropped = mc_lora.strip_networks(stage1_positive)
            inherited_negative, dropped_negative = mc_lora.strip_networks(stage1_negative)
            self._note_dropped_networks(dropped + dropped_negative)

            if mode == "Append":
                positive = ", ".join(
                    x for x in (inherited_positive.strip(), extra_positive.strip()) if x
                )
                negative = ", ".join(
                    x for x in (inherited_negative.strip(), extra_negative.strip()) if x
                )
            else:  # Inherit
                return inherited_positive, inherited_negative

        return mc_styles.apply(positive, negative, styles)

    def _note_dropped_networks(self, dropped) -> None:
        """Say once per generation which Stage 1 tags did not travel.

        Once, not once per image: a batch of eight would otherwise repeat the
        same line eight times, and the interesting fact is which tags were
        dropped, not how many prompts contained them.
        """
        if not dropped:
            return

        unique = sorted(set(dropped))
        if unique == self._dropped_networks:
            return

        self._dropped_networks = unique
        logger.info(
            "Model Chain: %s in the Stage 1 prompt %s not carried into Stage 2 — they are "
            "Stage-1-architecture-specific. Add a Stage 2 LoRA with <lora:name:weight> in "
            "Append or Replace mode.",
            ", ".join(unique),
            "was" if len(unique) == 1 else "were",
        )

    # -- hooks ------------------------------------------------------------- #

    def before_process(self, p, enabled, target, modules=None, *args):
        # Always reinstate a checkpoint we ourselves left swapped out, even when
        # the extension has since been disabled -- that is cleanup of our own
        # state, not extension behaviour.
        try:
            # A preload started after the last generation may still be running.
            # Joining first means the worst case is the wait we would have had
            # anyway, never two threads moving weights at once.
            mc_memory.join_preload()
            # Either branch means Stage 1 was swapped in from the cache; the
            # preload sized its VRAM budget from the *previous* generation, so
            # re-check it here against the size actually about to be sampled.
            swapped = mc_memory.reinstate_pending() or mc_memory.consume_preload()
            if swapped:
                self._make_room_for_stage_1(p)
            if swapped or enabled:
                self._report_readiness()
        except Exception:
            errors.report("Model Chain: failed to reinstate the cached checkpoint", exc_info=True)

        self._armed = False
        self._dropped_networks = []
        # Never carried between generations: the reference set describes the
        # gallery as it was for one job, and process() recaptures it for the next.
        self._references = ()
        # The divisor that turns the requested pixel size into a latent shape is
        # a host global rewritten by the loader, so it describes whichever
        # checkpoint was loaded last rather than the one that is loaded now. The
        # warm swap keeps it in step; this is the check that it is, made while
        # correcting it is still free.
        mc_memory.align_latent_scale()
        # After the reinstate above, so this describes the model Stage 1 is
        # about to sample with rather than whatever Stage 2 left loaded.
        self._stage1_geometry = mc_memory.describe_latent_geometry()
        self._stage1_requested = ()
        self._stage1_latent = ""

        if self._in_stage_2 or not enabled:
            return

        if not target or target == _NO_MODEL:
            logger.warning("Model Chain: enabled but no Stage 2 checkpoint is selected — skipping")
            return

        if mc_memory.checkpoint_info(target) is None:
            logger.warning('Model Chain: Stage 2 checkpoint "%s" was not found — skipping', target)
            p.comment(f'Model Chain: checkpoint "{target}" was not found; Stage 2 was skipped.')
            return

        # The built-in Refiner swaps the UNet mid-sampling and restores the
        # original weights in its own postprocess(), which runs after ours.
        # If we had swapped checkpoints by then it would write Model A's UNet
        # state dict into Model B. Refuse the combination rather than corrupt
        # the loaded model.
        if getattr(p, "refiner_checkpoint", None) not in (None, "", "None", "none"):
            message = (
                "Model Chain: the built-in Refiner is also enabled. The two cannot run "
                "together — Model Chain was skipped for this generation."
            )
            logger.warning(message)
            p.comment(message)
            return

        plan = mc_memory.plan(target, modules)
        logger.info("Model Chain: %s", plan.message)
        self._armed = True

    @staticmethod
    def _report_readiness() -> None:
        """Say how much of Stage 1 this generation starts with already in VRAM.

        The one line that makes the whole residency story checkable from the
        console: warm means the preload did its job, partially warm means it ran
        out of room, and cold means the next few seconds are model movement.

        Logged for chained generations and for the plain Generate that follows
        one -- the latter being the interesting case, since that is the
        generation the whole preload exists for. Not logged otherwise: someone
        who never enables Model Chain should never see a line from it.
        """
        try:
            _, message = mc_memory.stage_1_readiness()
        except Exception:
            return

        logger.info("%s", message)

    @staticmethod
    def _make_room_for_stage_1(p) -> None:
        """Give Stage 1 a clean VRAM budget after swapping its model back in.

        The previous generation ended with Stage 2's model in VRAM, and a warm
        swap does not evict it -- so Stage 1 would load into whatever is left.
        The host then makes room the hard way, partially unloading in small
        chunks while it loads, and that path is dramatically slower than a
        clean load: measured on a Krea 2 -> Flux.2 chain, the same ~8 GB UNet
        took 11.5s squeezed into 900 MB of spare VRAM against 0.8s with 7 GB
        spare.

        Stage 2 has had this since the pass-size fix; Stage 1 needs it for the
        same reason. When both models genuinely fit, make_vram_room() checks
        first and does nothing.

        Freeing is all that happens here, and that boundary is deliberate.
        ``free_memory`` moves weights *out* to their offload device; loading
        them *in* rewrites and re-patches them, and doing that from here -- a
        hook, not the sampler -- proved able to leave the model in a torch
        state the following sampling step rejects outright with
        ``RuntimeError: Inference tensors do not track version counter``.
        The host's own lazy load, driven from inside the sampler, is in the
        right context by construction. Nothing is gained by beating it to it.
        """
        try:
            width, height = mc_arch.stage1_size(p)
            mc_memory.make_vram_room(
                shared.opts.sd_model_checkpoint,
                mc_memory.current_modules(),
                width,
                height,
                stage="Stage 1",
            )
        except Exception:
            errors.report("Model Chain: failed to free VRAM for Stage 1", exc_info=True)

    def process(
        self,
        p,
        enabled,
        target,
        modules,
        prompt_mode,
        prompt,
        negative,
        styles,
        seed_mode,
        seed_offset,
        fixed_seed,
        cfg,
        steps,
        sampler,
        scheduler,
        denoise,
        size_multiplier,
        edit_mode=mc_arch.EDIT_AUTO,
        reference_mode=mc_references.DISABLED,
        reference_images=None,
        reference_max_dim=mc_references.DEFAULT_MAX_DIM,
        **kwargs,
    ):
        if not self._armed:
            return

        # Captured here, while Stage 1's script arguments are live: this is
        # where the user's ImageStitch gallery can still be read, and reading it
        # once means the Stage 2 set cannot drift if the panel changes while the
        # generation runs.
        self._references = self._capture_references(
            p, reference_mode, reference_images, reference_max_dim
        )

        # The requested size as the host settled it. process() runs after
        # process_images_inner has applied any model-driven adjustment to the
        # dimensions, and before the noise is built from them -- so this is the
        # last point at which the size is still just a number.
        self._stage1_requested = (
            int(getattr(p, "width", 0) or 0),
            int(getattr(p, "height", 0) or 0),
        )

        # Start the activation measurement here rather than in before_process:
        # this is the first point that is reached only for a chained generation,
        # and resetting the CUDA peak counters is the host's own instrumentation
        # to reset on every other one.
        mc_memory.begin_pass_observation()

        styles = list(styles or [])
        if prompt_mode == "Inherit":
            styles = []

        # Per-image resolved prompts. create_infotext indexes list values by
        # image index, so a prompt that varies across the batch is recorded
        # accurately for each image rather than collapsing to image 0's.
        total = len(p.all_prompts or [p.prompt])
        resolved_positive, resolved_negative = [], []
        for i in range(total):
            stage1_positive = p.all_prompts[i] if p.all_prompts else p.prompt
            stage1_negative = p.all_negative_prompts[i] if p.all_negative_prompts else p.negative_prompt
            pos, neg = self._resolve_prompts(
                stage1_positive, stage1_negative, prompt_mode, prompt, negative, styles
            )
            resolved_positive.append(pos)
            resolved_negative.append(neg)

        p.extra_generation_params.update(
            mc_infotext.build_params(
                target=target,
                prompt_mode=prompt_mode,
                prompt=resolved_positive,
                negative=resolved_negative,
                styles=styles,
                seed_mode=seed_mode,
                seed_offset=seed_offset,
                fixed_seed=fixed_seed,
                cfg=cfg,
                steps=steps,
                sampler=sampler,
                scheduler=scheduler,
                denoise=denoise,
                size_multiplier=size_multiplier,
                stage1_size="%dx%d" % mc_arch.stage1_size(p),
                edit_mode=edit_mode,
                modules=modules,
                reference_mode=reference_mode,
                reference_count=len(self._references[0]) if self._references else 0,
                reference_max_dim=reference_max_dim,
            )
        )

        # Stage 1 output is an intermediate: keep it out of the gallery and out
        # of the normal output folder (section 3.4). Saving is taken over by
        # postprocess() so the refined images carry the right infotext.
        self._save_final_images = p.save_samples()
        p.do_not_save_samples = True
        p.do_not_save_grid = True

    @staticmethod
    def _capture_references(p, mode, gallery, max_dim):
        """This job's supplemental reference set: ``(images, max_dim, reason)``.

        ``reason`` is empty when there is nothing to say. When it is not, it
        explains why the selected mode produced no references -- always a
        non-fatal condition, and always worth naming, because a mode that
        silently does nothing is indistinguishable from a broken one.
        """
        if not mc_references.is_active(mode):
            return (), max_dim, ""

        if mode == mc_references.PASS_THROUGH:
            arguments = mc_references.stitch_arguments(p)
            if arguments is None:
                return (), max_dim, "ImageStitch was not found, so there is nothing to pass through"

            enabled, stitch_gallery, stitch_max_dim = arguments
            if not enabled:
                return (), stitch_max_dim, "ImageStitch is switched off, so it has no references to pass through"

            images = mc_references.extract_images(stitch_gallery)
            if not images:
                return (), stitch_max_dim, "ImageStitch has no reference images selected"
            return tuple(images), stitch_max_dim, ""

        images = mc_references.extract_images(gallery)
        if not images:
            return (), max_dim, "no Stage 2 reference images have been added"
        return tuple(images), max_dim, ""

    def process_before_every_sampling(self, p, *args, **kwargs):
        """Record the latent Stage 1 is about to sample from.

        The one moment the requested size has stopped being a number and become
        a tensor. p.width/p.height say what was asked for and the finished image
        says what arrived; when those two disagree this is the measurement in
        between, and it says which side of the noise build lost the size.

        Stage 2 runs with scripts unset, so this only ever describes Stage 1.
        """
        if not self._armed or self._in_stage_2:
            return
        try:
            shape = getattr(kwargs.get("noise"), "shape", None)
            if shape:
                self._stage1_latent = "x".join(str(int(dim)) for dim in shape)
        except Exception:
            pass

    # -- Stage 2 ----------------------------------------------------------- #

    def postprocess(
        self,
        p,
        processed,
        enabled,
        target,
        modules,
        prompt_mode,
        prompt,
        negative,
        styles,
        seed_mode,
        seed_offset,
        fixed_seed,
        cfg,
        steps,
        sampler,
        scheduler,
        denoise,
        size_multiplier,
        edit_mode=mc_arch.EDIT_AUTO,
        reference_mode=mc_references.DISABLED,
        reference_images=None,
        reference_max_dim=mc_references.DEFAULT_MAX_DIM,
        **kwargs,
    ):
        if not self._armed or self._in_stage_2:
            return

        # Refine each result set once. postprocess() is normally called once per
        # generation, but a wrapper extension that retries a failed pass -- or
        # any other second invocation -- would otherwise re-refine images that
        # have already been through Stage 2, doubling the work and compounding
        # the effect on the output.
        if getattr(processed, _REFINED_MARKER, False):
            logger.info("Model Chain: this result set was already refined; skipping Stage 2")
            return
        setattr(processed, _REFINED_MARKER, True)

        self._armed = False

        try:
            self._run_stage_2(
                p,
                processed,
                target=target,
                modules=modules,
                prompt_mode=prompt_mode,
                extra_positive=prompt,
                extra_negative=negative,
                styles=list(styles or []),
                seed_mode=seed_mode,
                seed_offset=int(seed_offset or 0),
                fixed_seed=int(fixed_seed if fixed_seed is not None else -1),
                cfg=float(cfg),
                steps=int(steps),
                sampler=sampler,
                scheduler=scheduler,
                denoise=float(denoise),
                size_multiplier=float(size_multiplier),
                edit_mode=edit_mode,
                reference_mode=reference_mode,
            )
        except Exception:
            errors.report("Model Chain: Stage 2 failed; returning the unrefined Stage 1 images", exc_info=True)
            processed.comments += "\nModel Chain: Stage 2 failed — these images are unrefined Stage 1 output."
            self._save_as_final(p, processed)

    def _collect_stage_1(self, processed) -> list:
        """The N real Stage 1 images, excluding any grid."""
        start = processed.index_of_first_image
        count = len(processed.all_seeds)
        return list(processed.images[start : start + count])

    def _run_stage_2(
        self,
        p,
        processed,
        *,
        target,
        modules,
        prompt_mode,
        extra_positive,
        extra_negative,
        styles,
        seed_mode,
        seed_offset,
        fixed_seed,
        cfg,
        steps,
        sampler,
        scheduler,
        denoise,
        size_multiplier,
        edit_mode,
        reference_mode=mc_references.DISABLED,
    ):
        stage1_images = self._collect_stage_1(processed)
        if not stage1_images:
            return

        # Everything downstream sizes itself from the image Stage 2 was actually
        # handed, never from p.width/p.height. Those describe what was *asked
        # for*: with hires fix they name the first pass, and the host adjusts
        # them in process_images_inner before sampling -- so they are a
        # prediction, and this is the observation. Where the two disagree the
        # observation is the only one that describes a real image.
        stage1_width, stage1_height = stage1_images[0].width, stage1_images[0].height
        predicted = mc_arch.stage1_size(p)
        if (stage1_width, stage1_height) != predicted:
            # Warned, not noted: Stage 1 sampling at a size nobody asked for is
            # a fault in its own right, and Stage 2 faithfully preserving it
            # makes the extension look like the culprit. The geometry state is
            # carried along because it is what decided the size, and it is no
            # longer inspectable by the time anyone reads this.
            settled = (
                "%dx%d" % self._stage1_requested if self._stage1_requested else "not captured"
            )
            logger.warning(
                "Model Chain: Stage 1 produced %dx%d where %dx%d was requested — Stage 2 "
                "refines the image it was given, so the final output will be %dx%d. "
                "Sizes along the way: settled at %s, sampled from a %s latent. "
                "Stage 1's model state was: %s",
                stage1_width,
                stage1_height,
                *predicted,
                stage1_width,
                stage1_height,
                settled,
                self._stage1_latent or "not captured",
                self._stage1_geometry or "not captured",
            )
            processed.comments += (
                f"\nModel Chain: Stage 1 produced {stage1_width}x{stage1_height}, not the "
                f"{predicted[0]}x{predicted[1]} that was requested — the size was already "
                "lost before Stage 2 ran."
            )

        # Recorded now that it can be observed. process() had to write this
        # before Stage 1 ran, so it could only record the prediction.
        p.extra_generation_params[mc_infotext.STAGE1_SIZE] = f"{stage1_width}x{stage1_height}"

        # Stage 1 has finished sampling and Stage 2's model is not loaded yet,
        # so this is the one moment the peak reading describes Stage 1's pass
        # alone. It feeds the automatic VRAM reserve.
        mc_memory.observe_activation_peak(stage1_width, stage1_height, stage=mc_memory.STAGE_1)

        if prompt_mode == "Inherit":
            styles = []

        # Interrupted during Stage 1: abort before any model switch and hand
        # back what Stage 1 produced, clearly labelled (section 3.5).
        if state.interrupted or state.stopping_generation:
            logger.info("Model Chain: interrupted during Stage 1 — no model switch performed")
            processed.comments += "\nModel Chain: interrupted during Stage 1 — these images are unrefined."
            self._save_as_final(p, processed)
            return

        arch = mc_arch.detect_from_checkpoint_name(target)

        # Settled before the infotexts are built, so the reference count they
        # record is the set Stage 2 is actually given rather than the set that
        # was offered.
        #
        # Supplying references means asking for reference conditioning, so the
        # edit toggle is implied on for the Stage 2 pass. An explicit Disable is
        # the one thing that overrules it -- a control the user set by hand must
        # not be silently reversed by another one, so the references are dropped
        # and said out loud instead.
        references, reference_max_dim, reference_reason = (
            self._references if self._references else ((), mc_references.DEFAULT_MAX_DIM, "")
        )
        references = self._vet_references(
            processed, arch, references, reference_mode, reference_reason, edit_mode
        )
        if mc_references.is_active(reference_mode):
            p.extra_generation_params[mc_infotext.REFERENCE_COUNT] = len(references)

        effective_edit_mode = mc_arch.EDIT_ENABLE if references else edit_mode
        edit_override = mc_arch.edit_override(arch, effective_edit_mode)

        # Build every Stage 2 infotext now, while Model A is still loaded:
        # create_infotext reads live model attributes, and after the switch
        # those would describe Model B.
        infotexts = [
            create_infotext(p, processed.all_prompts, processed.all_seeds, processed.all_subseeds, index=i)
            for i in range(len(stage1_images))
        ]

        if shared.opts.model_chain_save_stage1:
            self._save_stage_1(p, processed, stage1_images, infotexts)

        # -- the one and only checkpoint switch (section 3.1) -------------- #
        # Both halves of Stage 1's pair are captured: restoring the checkpoint
        # without its VAE/text encoder would change what Stage 1 loads next time.
        previous_checkpoint = shared.opts.sd_model_checkpoint
        previous_modules = mc_memory.current_modules()
        # Captured while Model A is still loaded: its text encoder and VAE are
        # small enough to leave in VRAM through the switch, and moving them is a
        # disproportionate share of every switch's cost.
        mc_memory.capture_stage_1_encoders()

        started = time.perf_counter()
        transition = mc_memory.ensure_resident(target, modules)
        elapsed = time.perf_counter() - started

        logger.info(
            "Model Chain: switched to %s%s in %.1fs (%s) for %d image%s",
            target,
            self._describe_modules(modules),
            elapsed,
            _TRANSITION_DESCRIPTIONS.get(transition, transition),
            len(stage1_images),
            "" if len(stage1_images) == 1 else "s",
        )
        if transition == "cold":
            # Distinguish the one unavoidable first load from a model the cache
            # keeps refusing -- only the latter is something the user can act on.
            refused = mc_memory.last_refusal()
            if refused:
                logger.warning(
                    "Model Chain: this was a disk load because %s could not be cached. "
                    'Raise "Model Chain: max system RAM for model cache (GB)" in Settings '
                    "to make later switches warm.",
                    refused,
                )
            else:
                logger.info(
                    "Model Chain: first load of %s this session; later switches to it "
                    "should be warm swaps from the RAM cache.",
                    target,
                )

        state.job_count = (state.job_count or 0) + len(stage1_images)
        self._extend_progress_total(len(stage1_images) * int(steps))

        # Every image in the batch is refined at the same size, so one look at
        # the first is enough to size the pass -- and this has to happen before
        # the loop, while there is still something to evict.
        stage_2_width, stage_2_height = mc_arch.scaled_size(
            stage1_width, stage1_height, size_multiplier, arch.alignment
        )
        # The whole size story on one line: what Stage 2 was handed, what it was
        # asked to produce, and why the numbers land where they do. Without this
        # a size that comes out wrong gives the user nothing to look at.
        logger.info(
            "Model Chain: Stage 2 refines %dx%d at %.2fx — requesting %dx%d, aligned to %dpx for %s",
            stage1_width,
            stage1_height,
            size_multiplier,
            stage_2_width,
            stage_2_height,
            arch.alignment,
            arch.label,
        )
        mc_memory.make_vram_room(target, modules, stage_2_width, stage_2_height)
        mc_memory.begin_pass_observation()

        # A cached model keeps its engine object -- and its reference list --
        # across generations, and ImageStitch cannot clear it because Stage 2
        # runs with scripts disabled. Anything still in there is from an earlier
        # job. Cleared for every chain rather than only for an edit-mode one:
        # when both stages share a checkpoint the object is the *same* one
        # ImageStitch just filled for Stage 1, and Disabled mode promises Stage 2
        # sees none of it.
        self._clear_references(p)

        # Everything above worked from the checkpoint header, which is a guess
        # made before the model existed. The model is here now, so ask it.
        #
        # Kept in a separate name rather than replacing ``arch``: the geometry
        # above is already committed to the header's answer, and the panel
        # predicted the output size from that same answer. Correcting the size
        # here would silently disagree with both.
        reference_arch = arch
        if arch is mc_arch.UNKNOWN:
            loaded = mc_arch.detect_loaded_engine()
            if loaded is not mc_arch.UNKNOWN:
                reference_arch = loaded
                # The override has to be rebuilt: the empty one computed from
                # "unknown" would leave an opt-in architecture's toggle off, and
                # the references would encode into nothing.
                edit_override = mc_arch.edit_override(loaded, effective_edit_mode)
                logger.info(
                    "Model Chain: %s was not identifiable from its header, but the loaded "
                    "model reports it as %s — using that for reference support",
                    target,
                    loaded.label,
                )

        # Model B is loaded now, so its edit-mode state can finally be checked
        # against reality rather than against the dropdown's guess.
        edit_active = mc_arch.edit_is_active(reference_arch, effective_edit_mode)
        if edit_active:
            self._prepare_edit_mode(p, processed, reference_arch, denoise, edit_override, edit_mode)

        if references:
            # The infotexts are already written, so a set dropped here keeps the
            # pre-flight count in them; the notice on the result is what says it
            # was dropped. This still corrects ``p`` itself, so anything built
            # from it afterwards describes what happened rather than what was
            # planned.
            p.extra_generation_params[mc_infotext.REFERENCE_COUNT] = self._encode_references(
                p, processed, reference_arch, references, reference_max_dim, edit_override
            )

        refined: list = []
        delivered: list = []
        self._in_stage_2 = True
        try:
            for index, image in enumerate(stage1_images):
                if state.interrupted or state.stopping_generation:
                    break
                if state.skipped:
                    state.skipped = False

                state.job = f"Model Chain refine {index + 1}/{len(stage1_images)}"

                seed = self._stage_2_seed(processed.all_seeds[index], seed_mode, seed_offset, fixed_seed)
                positive, negative = self._resolve_prompts(
                    processed.all_prompts[index],
                    processed.all_negative_prompts[index],
                    prompt_mode,
                    extra_positive,
                    extra_negative,
                    styles,
                )
                width, height = mc_arch.scaled_size(
                    image.width, image.height, size_multiplier, arch.alignment
                )

                result = self._refine_one(
                    p,
                    image=image,
                    positive=positive,
                    negative=negative,
                    seed=seed,
                    width=width,
                    height=height,
                    cfg=cfg,
                    steps=steps,
                    sampler=sampler,
                    scheduler=scheduler,
                    denoise=denoise,
                    override_settings=dict(edit_override),
                )
                if result is None:
                    refined.append(image)
                    continue

                # The pass is what finally decides the output size, and it is
                # not obliged to honour the request: the host adjusts the
                # requested dimensions for some architectures, and any other
                # alwayson script can resize the result. Silently shipping a
                # size nobody asked for is what makes this class of bug so hard
                # to place, so it is recorded and reported instead.
                delivered.append((result.width, result.height))
                refined.append(result)
        finally:
            self._in_stage_2 = False
            # Before the selection moves back, while shared.sd_model is still
            # Stage 2's. Completion or failure alike: the cache keeps this model
            # object, so anything left on it belongs to no job at all by the time
            # it is next used.
            self._clear_references(p)
            mc_memory.observe_activation_peak(
                stage_2_width, stage_2_height, stage=mc_memory.STAGE_2
            )
            # Put the selection back so the next generation starts on Model A
            # with Model A's own VAE and text encoder. Only the selection:
            # reloading here would be a second switch.
            mc_memory.restore_selection(previous_checkpoint, previous_modules)
            self._preload_stage_1(p)

        self._report_size(processed, (stage_2_width, stage_2_height), delivered)
        self._finish(p, processed, stage1_images, refined, infotexts)

    @staticmethod
    def _report_size(processed, requested: tuple[int, int], delivered: list) -> None:
        """Say so when Stage 2 did not return the size it was asked for.

        The extension controls what it *requests* -- the Stage 1 image's size
        times the multiplier, snapped to the architecture's grid -- but not what
        the pass hands back. When those differ the user sees an unexplained
        resolution and has no way to tell which half of the chain lost it, so
        the mismatch is named in the console and on the result.
        """
        unexpected = sorted({size for size in delivered if size != requested})
        if not unexpected:
            return

        message = (
            "Model Chain: Stage 2 was asked for {}x{} but returned {}. The refine pass "
            "did not honour the requested size — check whether another extension resizes "
            "img2img output, and whether the Stage 2 architecture constrains its own "
            "resolution.".format(
                requested[0],
                requested[1],
                ", ".join(f"{w}x{h}" for w, h in unexpected),
            )
        )
        logger.warning(message)
        processed.comments += f"\n{message}"

    @staticmethod
    def _preload_stage_1(p) -> None:
        """Start warming Stage 1 back into VRAM for the next generation.

        This runs in the background precisely so it overlaps the time the user
        spends looking at the images they just got. Done inline it would gain
        nothing: postprocess() runs before the generation call returns, so the
        gallery would simply appear later by however long the load took.

        Nothing downstream depends on it finishing. If it fails, is still
        running, or never starts, the next generation loads Stage 1 exactly as
        it does today -- before_process() joins the thread and then does the
        work itself if it was not already done.
        """
        try:
            width, height = mc_arch.stage1_size(p)
            mc_memory.preload_async(width, height)
        except Exception:
            errors.report("Model Chain: failed to start the Stage 1 preload", exc_info=True)

    @staticmethod
    def _extend_progress_total(extra_steps: int) -> None:
        """Add Stage 2's steps to the "Total progress" bar.

        Stage 1 sizes that bar for its own passes only -- and hires fix resizes
        it again for its second pass -- so Stage 2's steps would overflow it,
        making the bar wrap and re-render as though the work were repeating.
        Purely cosmetic, hence best-effort.
        """
        if extra_steps <= 0:
            return
        try:
            bar = shared.total_tqdm._tqdm
            if bar is not None and bar.total:
                shared.total_tqdm.updateTotal(bar.total + extra_steps)
        except Exception:
            pass

    @staticmethod
    def _describe_modules(modules) -> str:
        """Short console description of the Stage 2 VAE / text encoder choice."""
        resolved = mc_memory.resolve_modules(modules)
        if resolved is None:
            return ""
        if not resolved:
            return " (built-in VAE/TE)"
        names = ", ".join(os.path.basename(m) for m in resolved)
        return f" (VAE/TE: {names})"

    def _stage_2_seed(self, stage1_seed: int, mode: str, offset: int, fixed: int) -> int:
        """Per-image seed for the refine pass (section 6.3).

        Inheriting each image's own Stage 1 seed is the default and is what the
        user gets without touching anything.
        """
        if mode == "Fixed":
            return int(fixed)
        if mode == "Offset":
            return int(stage1_seed) + int(offset)
        return int(stage1_seed)

    @staticmethod
    def _clear_references(p) -> None:
        """Drop reference state, and make ImageStitch re-encode its own next time.

        Both halves matter. The first stops an earlier job's references reaching
        this one; the second stops this clear from silently costing Stage 1 its
        references on the *next* generation, because ImageStitch memoises on the
        model and the image hashes and would otherwise skip an encode it needs.
        """
        mc_memory.clear_references()
        mc_references.invalidate_stitch_cache(p)

    def _vet_references(self, processed, arch, references, mode, reason, edit_mode):
        """The supplemental set Stage 2 may actually use, with a notice if not.

        Everything checkable before the model loads is checked here: whether the
        mode produced any images at all, whether the architecture has a
        reference path, and whether the user has explicitly turned reference
        conditioning off. Each rejection is non-fatal and named -- a mode that
        quietly does nothing is indistinguishable from one that is broken.
        """
        if not mc_references.is_active(mode):
            return ()

        if not references:
            self._report_references(
                processed,
                f'Stage 2 reference mode is "{mode}" but {reason or "no reference images were supplied"} '
                "— Stage 2 will run with only its own Stage 1 image.",
            )
            return ()

        if arch is mc_arch.UNKNOWN:
            # Not a rejection. The checkpoint's header did not identify it --
            # a quantised or repacked build, or a format the header reader
            # cannot open at all -- and a guess made before the switch is the
            # worst possible thing to refuse on when the loaded model is about
            # to be able to answer properly. Deferred, not dropped.
            logger.info(
                "Model Chain: the Stage 2 checkpoint could not be identified from its header; "
                "its reference support will be settled from the loaded model instead"
            )
            return references

        if not arch.supports_references:
            self._report_references(
                processed,
                f"{len(references)} Stage 2 reference image(s) were supplied, but {arch.label} has "
                "no reference-conditioning path — they were ignored rather than fed to a model "
                "that cannot use them.",
            )
            return ()

        if edit_mode == mc_arch.EDIT_DISABLE:
            self._report_references(
                processed,
                f"{len(references)} Stage 2 reference image(s) were supplied, but Stage 2 edit mode "
                'is set to "Disable" — the explicit setting wins and the references were ignored.',
            )
            return ()

        return references

    def _encode_references(self, p, processed, arch, references, max_dim, edit_override) -> int:
        """Register the supplemental references on the loaded Stage 2 model.

        Returns how many the model took. Nothing partial is ever shipped: if the
        model took fewer than were offered, the ordering the user arranged no
        longer holds, so the whole set is dropped and the pass runs as a plain
        refine rather than on a reference list that means something else.
        """
        if not arch.supports_references:
            # Deferred here from before the switch, where the checkpoint could
            # not be identified at all. Now it has been asked directly and the
            # answer is still no.
            label = arch.label if arch is not mc_arch.UNKNOWN else "the loaded Stage 2 model"
            self._report_references(
                processed,
                f"{len(references)} Stage 2 reference image(s) were supplied, but {label} has no "
                "reference-conditioning path — they were ignored rather than fed to a model that "
                "cannot use them.",
            )
            return 0

        if not mc_arch.references_available(arch):
            # The architecture is reference-capable but this checkpoint is not
            # the variant that has the path -- a plain Flux.1 where Kontext was
            # assumed, or a Qwen-Image that is not the Edit build. This is the
            # check no amount of looking at the name can make.
            self._report_references(
                processed,
                f"the loaded Stage 2 checkpoint does not expose {arch.label}'s reference path "
                f"(no {arch.reference_flag!r} model flag), so its {len(references)} reference "
                "image(s) were ignored.",
            )
            return 0

        if not arch.references_are_validated:
            # Said once the architecture is settled, so a checkpoint the header
            # could not identify is judged on what it turned out to be.
            self._report_references(
                processed,
                f"Stage 2 reference routing on {arch.label} is experimental — Forge exposes a "
                "reference path for it, but this pairing is not one Model Chain has validated.",
            )

        try:
            with mc_references.conditioning_enabled(edit_override):
                encoded = mc_references.encode(list(references), max_dim)
        except Exception:
            errors.report("Model Chain: failed to encode the Stage 2 reference images", exc_info=True)
            self._clear_references(p)
            self._report_references(
                processed,
                "the Stage 2 reference images could not be encoded — the refine ran without them.",
            )
            return 0

        if encoded != len(references):
            self._clear_references(p)
            self._report_references(
                processed,
                f"the Stage 2 model took {encoded} of {len(references)} reference image(s), so the "
                "order you arranged no longer holds — the whole set was dropped rather than used "
                "incomplete.",
            )
            return 0

        logger.info(
            "Model Chain: %d supplemental reference image(s) registered for every Stage 2 image, "
            "after each image's own Stage 1 handoff",
            encoded,
        )
        return encoded

    @staticmethod
    def _report_references(processed, message: str) -> None:
        """Say something about references in the console and on the result."""
        text = f"Model Chain: {message}"
        logger.warning(text)
        processed.comments += f"\n{text}"

    def _prepare_edit_mode(self, p, processed, arch, denoise, edit_override, mode):
        """Sanity-check reference conditioning for the Stage 2 pass.

        Stale reference state is cleared by the caller, for every chain rather
        than only an edit-mode one.
        """
        logger.info(
            "Model Chain: %s reference conditioning active%s",
            arch.label,
            f" ({arch.edit_option} overridden for Stage 2)" if edit_override else "",
        )

        if not arch.edit_is_deliberate(mode):
            # Klein left on its default: references are simply how it works, and
            # an ordinary low-denoise refine is valid. Warning here would fire on
            # every Klein chain and assert something the host itself does not.
            return

        if denoise < arch.edit_denoise - 0.1:
            message = (
                f"Model Chain: {arch.label} edit mode uses the Stage 1 image as a "
                f"reference rather than a starting image, so denoise {denoise:g} is "
                f"low — {arch.edit_denoise:g} is the expected value."
            )
            logger.warning(message)
            processed.comments += f"\n{message}"

        if arch.edit_needs_lora and not self._stage_2_has_lora(p):
            message = (
                f"Model Chain: {arch.label} edit mode needs its Edit LoRA, but no "
                "extra-network tag was found in the Stage 2 prompt. Add it with "
                "<lora:name:weight> in Append or Replace mode."
            )
            logger.warning(message)
            processed.comments += f"\n{message}"

    @staticmethod
    def _stage_2_has_lora(p) -> bool:
        """Whether the resolved Stage 2 prompt carries any extra-network tag.

        Deliberately a presence check, not a parse: the extension must not
        interpret extra-network syntax itself (section 5.4). It only needs to
        know whether the user supplied one at all.
        """
        recorded = p.extra_generation_params.get(mc_infotext.PROMPT, "")
        prompts = recorded if isinstance(recorded, list) else [recorded]
        return any("<lora:" in str(text) or "<lyco:" in str(text) for text in prompts)

    def _refine_one(
        self, p, *, image, positive, negative, seed, width, height, cfg, steps, sampler,
        scheduler, denoise, override_settings=None,
    ):
        p2 = StableDiffusionProcessingImg2Img(
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=positive,
            negative_prompt=negative,
            # Styles are already expanded into the prompt above; passing them
            # again here would apply them twice.
            styles=[],
            seed=seed,
            subseed=-1,
            sampler_name=p.sampler_name if sampler == mc_infotext.INHERIT else sampler,
            scheduler=p.scheduler if scheduler == mc_infotext.INHERIT else scheduler,
            batch_size=1,
            n_iter=1,
            steps=steps,
            cfg_scale=cfg,
            width=width,
            height=height,
            init_images=[image],
            denoising_strength=denoise,
            resize_mode=0,
            # Reference/edit conditioning is gated by a *global* Settings
            # toggle. Routing it through override_settings scopes it to this
            # generation and lets the host restore the user's value afterwards,
            # rather than leaving it flipped for everything they do next.
            override_settings=override_settings or {},
        )

        # Stage 2 is an independent generation, not a nested script run: no
        # alwayson script from the Stage 1 pass carries over. Extra networks are
        # unaffected by this -- they are handled by the core processing loop,
        # which is exactly how <lora:...> tags in the Stage 2 prompt get parsed,
        # applied against Model B, and deactivated afterwards.
        p2.scripts = None
        p2.script_args = []

        # Stage 2 is a single img2img pass over whatever Stage 1 produced.
        # If Stage 1 used hires fix, that upscale is already baked into the
        # image; re-running it here would upscale twice and cost a second
        # sampling pass. img2img has no hires fix of its own, so this is
        # belt and braces against anything copying the flag across.
        p2.enable_hr = False

        # This script writes the final files itself, using Stage-1-derived
        # infotext, so the inner call must not save anything.
        p2.do_not_save_samples = True
        p2.do_not_save_grid = True

        try:
            result = process_images(p2)
        except Exception:
            errors.report("Model Chain: a Stage 2 refine pass failed", exc_info=True)
            # The pass may have failed *during* LoRA application, which would
            # leave the host believing a state is applied that partly is not.
            # This model is cached rather than reloaded, so that belief would
            # outlive the job that created it and quietly affect every later
            # generation on Model B. Throwing it away costs one reapplication.
            mc_memory.invalidate_prepared_state("a Stage 2 refine pass failed")
            return None
        finally:
            p2.close()

        if not result.images:
            return None
        return result.images[result.index_of_first_image]

    # -- output ------------------------------------------------------------ #

    def _finish(self, p, processed, stage1_images, refined, infotexts):
        """Assemble the final result set and write it to disk."""
        count = len(stage1_images)
        mixed = len(refined) < count

        if mixed:
            # Never silently return a short batch (section 3.5): pad with the
            # unrefined remainder and say so.
            missing = count - len(refined)
            refined = refined + stage1_images[len(refined) :]
            message = (
                f"Model Chain: interrupted during Stage 2 — {len(refined) - missing} of {count} "
                f"images were refined, the remaining {missing} are unrefined Stage 1 output."
            )
            logger.warning(message)
            processed.comments += f"\n{message}"

        for index, image in enumerate(refined):
            info = infotexts[index]
            if opts.enable_pnginfo:
                image.info["parameters"] = info
            if self._save_final_images:
                images.save_image(
                    image,
                    p.outpath_samples,
                    "",
                    processed.all_seeds[index],
                    processed.all_prompts[index],
                    opts.samples_format,
                    info=info,
                    p=p,
                )

        start = processed.index_of_first_image
        processed.images[start : start + count] = refined
        processed.infotexts[start : start + count] = infotexts
        if processed.infotexts:
            processed.info = processed.infotexts[start] if len(processed.infotexts) > start else infotexts[0]

    def _save_as_final(self, p, processed):
        """Write Stage 1 images out as the final result of an aborted chain."""
        if not self._save_final_images:
            return

        start = processed.index_of_first_image
        for index in range(len(processed.all_seeds)):
            position = start + index
            if position >= len(processed.images):
                break
            info = processed.infotexts[position] if position < len(processed.infotexts) else ""
            try:
                images.save_image(
                    processed.images[position],
                    p.outpath_samples,
                    "",
                    processed.all_seeds[index],
                    processed.all_prompts[index],
                    opts.samples_format,
                    info=info,
                    p=p,
                )
            except Exception:
                errors.report("Model Chain: failed to save a Stage 1 image", exc_info=True)

    def _save_stage_1(self, p, processed, stage1_images, infotexts):
        """Write Stage 1 intermediates to their own subfolder (section 3.4)."""
        directory = os.path.join(p.outpath_samples, STAGE1_SUBFOLDER)
        for index, image in enumerate(stage1_images):
            try:
                images.save_image(
                    image,
                    directory,
                    "",
                    processed.all_seeds[index],
                    processed.all_prompts[index],
                    opts.samples_format,
                    info=infotexts[index],
                    p=p,
                )
            except Exception:
                errors.report("Model Chain: failed to save a Stage 1 intermediate", exc_info=True)


def _on_script_unloaded():
    mc_memory.release_all()


try:
    from modules import script_callbacks

    script_callbacks.on_script_unloaded(_on_script_unloaded)
except Exception:
    errors.report("Model Chain: failed to register the unload callback", exc_info=True)
