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
import mc_memory
import mc_presets
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

shared.options_templates.update(
    shared.options_section(
        (None, "Model Chain"),
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
        """Capture main-tab controls that drive the live size readout."""
        attribute = self._TRACKED_COMPONENTS.get(kwargs.get("elem_id"))
        if attribute is not None:
            setattr(self, attribute, component)

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
        ]

    # -- prompt assembly --------------------------------------------------- #

    def _resolve_prompts(self, stage1_positive, stage1_negative, mode, extra_positive, extra_negative, styles):
        """Assemble one Stage 2 prompt pair.

        Order is prompt assembly -> style expansion -> (host) extra-networks
        parsing, per section 5.4. Extra-network tags are never touched here:
        the assembled string is handed to the normal processing pipeline, which
        parses and strips them exactly as it does for the main prompt box.
        """
        if mode == "Replace":
            positive, negative = extra_positive, extra_negative
        elif mode == "Append":
            positive = ", ".join(x for x in (stage1_positive.strip(), extra_positive.strip()) if x)
            negative = ", ".join(x for x in (stage1_negative.strip(), extra_negative.strip()) if x)
        else:  # Inherit
            return stage1_positive, stage1_negative

        return mc_styles.apply(positive, negative, styles)

    # -- hooks ------------------------------------------------------------- #

    def before_process(self, p, enabled, target, modules=None, *args):
        # Always reinstate a checkpoint we ourselves left swapped out, even when
        # the extension has since been disabled -- that is cleanup of our own
        # state, not extension behaviour.
        try:
            mc_memory.reinstate_pending()
        except Exception:
            errors.report("Model Chain: failed to reinstate the cached checkpoint", exc_info=True)

        self._armed = False

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
        **kwargs,
    ):
        if not self._armed:
            return

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
            )
        )

        # Stage 1 output is an intermediate: keep it out of the gallery and out
        # of the normal output folder (section 3.4). Saving is taken over by
        # postprocess() so the refined images carry the right infotext.
        self._save_final_images = p.save_samples()
        p.do_not_save_samples = True
        p.do_not_save_grid = True

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
    ):
        stage1_images = self._collect_stage_1(processed)
        if not stage1_images:
            return

        if prompt_mode == "Inherit":
            styles = []

        # Interrupted during Stage 1: abort before any model switch and hand
        # back what Stage 1 produced, clearly labelled (section 3.5).
        if state.interrupted or state.stopping_generation:
            logger.info("Model Chain: interrupted during Stage 1 — no model switch performed")
            processed.comments += "\nModel Chain: interrupted during Stage 1 — these images are unrefined."
            self._save_as_final(p, processed)
            return

        # Build every Stage 2 infotext now, while Model A is still loaded:
        # create_infotext reads live model attributes, and after the switch
        # those would describe Model B.
        infotexts = [
            create_infotext(p, processed.all_prompts, processed.all_seeds, processed.all_subseeds, index=i)
            for i in range(len(stage1_images))
        ]

        if shared.opts.model_chain_save_stage1:
            self._save_stage_1(p, processed, stage1_images, infotexts)

        arch = mc_arch.detect_from_checkpoint_name(target)
        edit_override = mc_arch.edit_override(arch, edit_mode)

        # -- the one and only checkpoint switch (section 3.1) -------------- #
        # Both halves of Stage 1's pair are captured: restoring the checkpoint
        # without its VAE/text encoder would change what Stage 1 loads next time.
        previous_checkpoint = shared.opts.sd_model_checkpoint
        previous_modules = mc_memory.current_modules()

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
        first = stage1_images[0]
        stage_2_width, stage_2_height = mc_arch.scaled_size(
            first.width, first.height, size_multiplier, arch.alignment
        )
        mc_memory.make_vram_room(target, modules, stage_2_width, stage_2_height)

        # Model B is loaded now, so its edit-mode state can finally be checked
        # against reality rather than against the dropdown's guess.
        edit_active = mc_arch.edit_is_active(arch, edit_mode)
        if edit_active:
            self._prepare_edit_mode(p, processed, arch, denoise, edit_override, edit_mode)

        refined: list = []
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
                refined.append(result if result is not None else image)
        finally:
            self._in_stage_2 = False
            # Put the selection back so the next generation starts on Model A
            # with Model A's own VAE and text encoder. Only the selection:
            # reloading here would be a second switch.
            mc_memory.restore_selection(previous_checkpoint, previous_modules)

        self._finish(p, processed, stage1_images, refined, infotexts)

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

    def _prepare_edit_mode(self, p, processed, arch, denoise, edit_override, mode):
        """Set up and sanity-check reference conditioning for the Stage 2 pass."""
        # A cached model keeps its engine object -- and its reference list --
        # across generations, and ImageStitch cannot clear it because Stage 2
        # runs with scripts disabled. Anything still in there is stale. This
        # applies whenever references will be used at all, deliberate or not.
        mc_memory.clear_references()

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
