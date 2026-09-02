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
import mc_arm
import mc_broker
import mc_hint
import mc_infotext
import mc_literal_report
import mc_llm_paths
import mc_llm_runtime
import mc_llm_state
import mc_llm_studio
import mc_logfile
import mc_lora
import mc_memory
import mc_pipeline_panel
import mc_plan
import mc_plan_panel
import mc_presets
import mc_profile_state
import mc_progress
import mc_references
import mc_styles
import mc_voice_api
import mc_voice_clone
import mc_voice_device
import mc_voice_engines
import mc_voice_models
import mc_voice_paths
import mc_voice_profile
import mc_voice_registry
import mc_voice_runtime
import mc_voice_sopro
import mc_voice_sopro_profile
import mc_voice_cleanup_runtime
import mc_voice_sopro_runtime
import mc_voice_pocket
import mc_voice_pocket_profile
import mc_voice_pocket_runtime
import mc_voice_pipeline
import mc_voice_pipeline_runtime
import mc_voice_state
import mc_voice_ui
from modules import errors, images, processing, scripts, shared
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from modules.shared import opts, state
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

logger = mc_memory.logger
"""Shared with the helper modules; mc_memory attaches the console handler."""

STAGE1_SUBFOLDER = "model-chain-stage1"

_GB = 1024**3


def _megapixels(width, height) -> float:
    """Pixel count in megapixels, the unit sampling rates are normalised by."""
    return max(int(width or 0) * int(height or 0), 0) / 1_000_000


def _warm_up_wanted() -> bool:
    """Whether the user has asked for models to be kept warm at all.

    The preload setting says whether a background thread may run after a
    generation; the warm-up setting says whether the pipeline should be loaded
    before somebody waits on it. They are different questions with different
    defaults, and a user who has answered yes to the second has said what they
    want about the first -- so keeping the image model on the card between
    generations follows from either, and needs both only if the aim is a
    warm-up that does nothing.
    """
    try:
        return mc_arm.mode() != mc_arm.WARM_OFF
    except Exception:
        return False


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

OPT_SMOOTH = "model_chain_smooth_progress"
"""Interpolate the host's stepped progress bar between polls.

Kept apart from the appearance settings below because it is not an appearance:
it does not change how the bar looks, only how often it is redrawn between two
widths the host already asked for. Default on, and independent of every other
toggle here -- installing the extension is the whole condition.
"""

OPT_HIDE_FOOTER = "model_chain_hide_footer"
"""Take the WebUI's footer off the page. Read by javascript/llm_studio.js.

Not an appearance setting and not really about this extension's own panels,
which is why it sits on its own. The footer is below the fold and takes real
space, so a page that otherwise fits the window still scrolls -- by exactly the
height of a row of links. LLM Studio's workspace is built to fit the window
precisely: the page does not scroll, the transcript does, and one scrollable
strip under it undoes that from outside anything this extension can lay out.

Default on, because the ask was *"can we just make the footer go away"* and the
footer's contents (the API link, the Github link, the Gradio credit, the version
line) are not things anybody reads twice. Off puts it straight back -- no
reload, because the rule is CSS keyed on an attribute this reads live.
"""

OPT_STYLE_ENABLE = "model_chain_style_enable"
OPT_STYLE_THEME = "model_chain_style_theme"
OPT_STYLE_COLOR = "model_chain_style_color"
OPT_STYLE_GRADIENT = "model_chain_style_gradient"
OPT_STYLE_SHEEN = "model_chain_style_sheen"
OPT_STYLE_GLOW = "model_chain_style_glow"
OPT_STYLE_COMPLETE = "model_chain_style_complete"
"""Appearance settings, read by javascript/model_chain_progress.js.

Every option registered here reaches the browser as a field of the global
``opts`` object, which is the whole mechanism: the styling layer needs no
endpoint, no Gradio component and no Python of its own. It is also why these
live in Settings rather than in the accordion -- a look is a property of the
install, not of one generation, and an accordion control would travel in presets
and infotexts where it means nothing.
"""

STYLE_CUSTOM = "Custom"

STYLE_THEMES = ("Flat", "Gradient", "Sheen", "Pulse", "Neon", "Ooze", STYLE_CUSTOM)
"""The ready-made looks, plus the one that defers to the toggles.

Each named theme is a choice of the four effects ``style.css`` implements --
gradient, sheen, glow, pulse -- made in the JS rather than here, so there is one
implementation of each effect rather than one per theme. This list exists only
to populate the dropdown; a test checks it against the JS so the two cannot
drift apart.

Every theme derives its colours from a single custom property, which is what
lets the colour setting recolour all of them rather than only the plain one.
"""

STYLE_THEME_DEFAULT = "Flat"


def _kv_types() -> list[tuple[str, str]]:
    """The key/value cache types llama.cpp offers, as ``(name, label)``.

    Read from ``mc_llm_context`` rather than listed here, and wrapped because
    this runs at import: a settings section that cannot be registered would
    take the whole extension down with it, and f16 alone is a correct -- if
    short -- list for a host where the LLM half will not import.
    """
    try:
        import mc_llm_context

        return [(name, label) for name, label in mc_llm_context.KV_TYPE_LABELS]
    except Exception:
        return [("f16", "f16")]


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
                "Keep the image model in VRAM between generations (experimental)",
            ).info(
                "moves the image model's weights back into VRAM in the background while "
                "you look at the result, so the next Generate starts sampling immediately "
                "— and so a LoRA is applied to weights already on the card rather than "
                "walked across the bus first. Off by default: it is the only part of this "
                "extension that touches models off the generation thread, and on some "
                "setups — on-the-fly LoRA patching in particular — that has broken the "
                "following generation outright. \"Warm up before generating\" does the same "
                "work at the moments it names, so turning that on turns this on for those "
                "moments too"
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
                "Requires a warm-up — the setting above, or \"Warm up before generating\": "
                "filling VRAM is only safe on that thread, so with neither asked for this "
                "does nothing"
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
            mc_progress.OPT_PROGRESS: shared.OptionInfo(
                mc_progress.PROGRESS_DEFAULT,
                "Predict progress and ETA for the whole chained job",
            ).info(
                "the host counts sampling steps, which makes the bar finish at the end of "
                "Stage 1 and then jump backwards. This models the job as timed phases "
                "instead — including the model switch — and reports how much of the "
                "predicted wall time has passed. Timings are measured on your machine and "
                "improve after a few generations"
            ),
            OPT_SMOOTH: shared.OptionInfo(
                True,
                "Advance the progress bar smoothly",
            ).info(
                "the WebUI writes the bar's width once per progress poll, so it arrives in "
                "steps — one jump every <em>Progress bar update period</em> milliseconds. "
                "This fills in the movement between those writes. It changes nothing about "
                "the numbers, applies to every generation whether or not Model Chain is "
                "involved, and is independent of the appearance settings below"
            ),
            OPT_HIDE_FOOTER: shared.OptionInfo(
                True,
                "Hide the WebUI footer",
            ).info(
                "the footer sits below the fold and takes real space, so a page that "
                "otherwise fits the window still scrolls by the height of a row of links. "
                "LLM Studio's conversation workspace is built to fit the window exactly, "
                "and this is the one strip under it that no layout here can reach. Applies "
                "to every tab, because the footer is one element on the page; turning it "
                "off puts it straight back"
            ),
            # -- appearance, deliberately independent of everything above --- #
            # These change how the host's bar looks and never what it reports,
            # so they apply to ordinary Forge generations too and stay useful
            # with Model Chain switched off entirely.
            OPT_STYLE_ENABLE: shared.OptionInfo(
                False,
                "Custom progress-bar appearance",
            ).info(
                "restyles the WebUI's own progress bar for every generation, chained or "
                "not. Purely cosmetic: it never changes the numbers on the bar"
            ),
            OPT_STYLE_THEME: shared.OptionInfo(
                STYLE_THEME_DEFAULT,
                "Progress-bar theme",
                gr.Dropdown,
                {"choices": list(STYLE_THEMES)},
            ).info(
                "<b>Flat</b> plain fill · <b>Gradient</b> lightens towards the leading edge · "
                "<b>Sheen</b> adds a travelling highlight · <b>Pulse</b> a breathing halo · "
                "<b>Neon</b> all of it, turned up · <b>Ooze</b> glowing sludge that fills "
                "from the left and bubbles, with bubbles escaping above the bar · "
                "<b>Custom</b> uses the three toggles below. Each one takes its colours from "
                "the setting underneath, so picking a colour recolours whichever theme you "
                "chose — <code>#39ff5e</code> makes Ooze toxic slime"
            ),
            OPT_STYLE_COLOR: shared.OptionInfo(
                "",
                "Progress-bar colour",
            ).info(
                "any CSS colour — <code>#38bdf8</code>, <code>rgba(56, 189, 248, 0.85)</code>, "
                "<code>hsl(199 89% 60%)</code>. Leave it empty to follow the active theme's "
                "accent colour, which is what keeps it looking right in light and dark and "
                "under third-party themes"
            ),
            OPT_STYLE_GRADIENT: shared.OptionInfo(
                False,
                "Custom theme: fade the fill towards its leading edge",
            ).info(f'ignored unless the theme above is "{STYLE_CUSTOM}"'),
            OPT_STYLE_SHEEN: shared.OptionInfo(
                False,
                "Custom theme: send a highlight travelling along the bar",
            ).info(
                f'ignored unless the theme above is "{STYLE_CUSTOM}". Stands down by itself '
                "when the active WebUI theme already animates the bar, so the two do not "
                "move over each other"
            ),
            OPT_STYLE_GLOW: shared.OptionInfo(
                False,
                "Custom theme: glow and pulse while generating",
            ).info(
                f'ignored unless the theme above is "{STYLE_CUSTOM}". Drawn outside the bar '
                "so it cannot make the percentage or ETA harder to read"
            ),
            OPT_STYLE_COMPLETE: shared.OptionInfo(
                False,
                "Flash the bar when a job finishes",
            ).info(
                "once, when the whole job is done — in a chained generation that is after "
                "Stage 2, never at the end of Stage 1. Applies to every theme"
            ),
            # -- LLM Studio and cross-workload residency ------------------- #
            #
            # These describe the machine and the policy for sharing it, not
            # the image being made, so they belong here rather than in the
            # accordion for the same reason the VRAM settings above do: a
            # residency strategy says nothing useful in an infotext.
            mc_llm_studio.OPT_ENABLE: shared.OptionInfo(
                True,
                "Show the LLM Studio tab",
            ).info(
                "the local-LLM workspace: LTX prompt generation, conversation and the "
                "MiniMax H3 enhancer. Turn it off to keep this extension to its image "
                "features — nothing about ordinary generation changes either way"
            ),
            mc_broker.OPT_MODE: shared.OptionInfo(
                mc_broker.MODE_HYBRID,
                "VRAM residency mode",
                gr.Radio,
                {"choices": [label for _, label in mc_broker.MODES]},
            ).info(
                "The image model always keeps its VRAM: the LLM is placed in what is left "
                "over and shrinks itself — shorter context, then experts and blocks in "
                "system RAM — rather than moving a checkpoint. That is not a setting. "
                "This is: whether a warm llama-server sitting in that spare room is left "
                "alone when a generation starts, or stopped so the pass has every last "
                "byte. Keeping it costs the generation nothing measurable and saves a "
                "model load — which on a slow placement is twenty seconds before the "
                "first word of the next prompt. On \"Free the LLM for every image\", the "
                "warm-up below leaves llama-server alone as well: it would only be "
                "starting one for the generation to stop, so it starts on the first "
                "request that actually needs it"
            ),
            mc_plan.OPT_CAP_MODE: shared.OptionInfo(
                mc_plan.CAP_AUTO,
                "Persistent LLM VRAM",
                gr.Radio,
                {"choices": [label for _, label in mc_plan.CAP_MODES]},
            ).info(
                "Auto sizes llama-server from what the generation's own plan leaves over — "
                "the largest single phase of it, not the sum of the phases, because Stage 1 "
                "and Stage 2 take over one another's VRAM rather than sharing the card. "
                "Custom puts a lower ceiling on that, which buys image headroom back at the "
                "cost of LLM tokens per second. Off keeps no persistent residency at all"
            ),
            mc_plan.OPT_CAP_GB: shared.OptionInfo(
                0.0,
                "Custom persistent LLM cap (GB)",
                gr.Number,
            ).info(
                "only read when the setting above is Custom. A figure above the calculated "
                "allowance does not raise it — this control is a way to be more "
                "conservative than the arithmetic, never less"
            ),
            mc_plan.OPT_SAFETY_GB: shared.OptionInfo(
                0.0,
                "Extra image safety headroom (GB)",
                gr.Number,
            ).info(
                "added on top of the automatic reserve when the plan's budget is worked "
                "out, and therefore taken off what the LLM may hold. Raise it if a "
                "generation has ever had to evict llama-server to finish"
            ),
            mc_llm_runtime.OPT_RELEASE: shared.OptionInfo(
                mc_llm_runtime.RELEASE_STOP,
                "When the image side needs the LLM's VRAM back",
                gr.Radio,
                {"choices": [label for _, label in mc_llm_runtime.RELEASE_MODES]},
            ).info(
                "stopping llama-server releases every byte of it and leaves the weights in "
                "the system page cache, so restarting reads from RAM rather than disk. "
                "Keeping it running moves the model to system RAM instead: no reload, much "
                "slower generation, and the RAM stays spent"
            ),
            mc_arm.OPT_WARM_UP: shared.OptionInfo(
                mc_arm.WARM_OFF,
                "Warm up before generating",
                gr.Radio,
                {"choices": [label for _, label in mc_arm.WARM_MODES]},
            ).info(
                "a cold run and a warm one are the same work in a different order: the "
                "model load, the placement and the prompt cache all happen once and then "
                "stop happening. From one log, five identical jobs ran in 82s, 27s, 3.6s, "
                "4.1s and 5.1s. This decides when that is paid for — halfway through a "
                "generation somebody is watching, or before it starts. The image model is "
                "warmed first, because that is what Generate waits on; it loads off the "
                "generation thread, which is the caveat on the experimental setting above. "
                "With VRAM residency set to free the LLM for every image, llama-server is "
                "not warmed at all — a generation would stop it moments later"
            ),
            mc_llm_runtime.OPT_LLM_SLOTS: shared.OptionInfo(
                mc_llm_runtime.SLOTS_AUTOMATIC,
                "Warm prompt caches llama-server keeps",
                gr.Radio,
                {"choices": [label for _, label in mc_llm_runtime.SLOT_MODES]},
            ).info(
                "every mode opens with a different system prompt — Conversation, Prompt "
                "Studio, MiniMax, the Krea writer, the Spatial Composer — and with one cache "
                "between them each switch re-reads the prefix the last one had just cached. "
                "A cache each costs a key/value cache each and llama.cpp routes every prompt "
                "to the cache that matches it best. Automatic buys them only out of VRAM "
                "left over once the model already fits, so it can never be the reason a "
                "context shrank or a layer left the card; under pressure they are the first "
                "thing given back"
            ),
            mc_llm_runtime.OPT_ROLE_PROCESSES: shared.OptionInfo(
                mc_llm_runtime.PROCESSES_SHARED,
                "When Creative and Spatial are configured identically",
                gr.Radio,
                {"choices": [label for _, label in mc_llm_runtime.PROCESS_MODES]},
            ).info(
                "identical roles resolve to one llama-server by default: one process, one copy "
                "of the weights, one prompt cache. The cost is that the two passes use "
                "different system prompts on it, so each switch re-reads a prefix the other "
                "just cached. A card or a machine with room for two servers can have one each "
                "instead — both stay warm, and neither pass ever re-reads the other's prompt"
            ),
            mc_llm_runtime.OPT_ROLE_SHARING: shared.OptionInfo(
                mc_llm_runtime.SHARE_AUTO,
                "When Creative and Spatial want the same memory",
                gr.Radio,
                {"choices": [label for _, label in mc_llm_runtime.SHARING_MODES]},
            ).info(
                "only reached when the two roles are configured differently and still land in "
                "the same place. Configure them identically and they share one llama-server "
                "whatever this says; put them on different devices and they never meet. Two "
                "servers cannot share a process, so taking turns means stopping one before "
                "starting the other — a model load per role switch, in exchange for never "
                "competing. Coexisting leaves both up and lets the existing memory rules sort "
                "it out, which on one card means loading and unloading as they fight for what "
                "is left"
            ),
            mc_llm_paths.OPT_ROOT: shared.OptionInfo(
                "",
                "LLM data directory",
                gr.Textbox,
            ).info(
                "where LLM Studio keeps the runtime, the model, characters and chats. Empty "
                "uses a folder in your WebUI data directory. Point it at an existing Prompt "
                "Master install to reuse a model you have already downloaded"
            ),
            mc_llm_paths.OPT_MODELS: shared.OptionInfo(
                "",
                "LLM models folder",
                gr.Textbox,
            ).info(
                "scanned for .gguf files to fill the model chooser at the top of the LLM "
                "Studio tab, so a model can be switched to without typing a path. Empty scans "
                "the models folder inside the LLM data directory above. A model does not have "
                "to be in here to be used — it is read, not started — this is only what the "
                "chooser offers"
            ),
            # -- what the LLM is placed at ---------------------------------
            #
            # These five were controls in the tab's own "Models, hardware and
            # memory" panel and are here now for the same reason the VRAM
            # settings above are: every one of them describes the installation
            # rather than the click being made, the host already persists and
            # restores exactly this kind of value, and a context budget says
            # nothing useful in an infotext. The tab keeps what a Settings page
            # cannot draw -- the file dialogs, the download, the estimate and
            # the residency table -- in its Setup mode.
            mc_llm_state.OPT_CONTEXT_MODE: shared.OptionInfo(
                mc_llm_state.label_for_context_mode("auto"),
                "LLM context sizing",
                gr.Radio,
                {"choices": [label for _, label in mc_llm_state.CONTEXT_MODES]},
            ).info(
                "Automatic budgets the key/value cache from whatever VRAM is free once the "
                "weights and the reserve are accounted for, so a context grows to fill an "
                "empty card and shrinks when a checkpoint arrives. Fixed buffer spends the "
                "number below and no more"
            ),
            mc_llm_state.OPT_CONTEXT_BUFFER: shared.OptionInfo(
                4.0,
                "LLM context / VRAM buffer (GB)",
                gr.Number,
            ).info(
                "memory budgeted for the key/value cache, separate from the weights and from "
                "the runtime reserve. Used when sizing is Fixed buffer"
            ),
            mc_llm_state.OPT_CONTEXT_SIZE: shared.OptionInfo(
                8192,
                "LLM context size (tokens)",
                gr.Number,
            ).info(
                "used when sizing is Fixed buffer, and never allowed past the model's own "
                "ceiling. A number rather than a slider because modern ceilings run to a "
                "million tokens and no slider across that range can be aimed at 32,768"
            ),
            mc_llm_state.OPT_KV_TYPE_K: shared.OptionInfo(
                "f16",
                "LLM K cache type",
                gr.Dropdown,
                {"choices": [name for name, _ in _kv_types()]},
            ).info(
                "quantising the key cache buys context at some cost in reply quality. "
                "llama.cpp sizes the two halves separately, which is why there are two "
                "settings; f16 for both is what the estimator's numbers assume"
            ),
            mc_llm_state.OPT_KV_TYPE_V: shared.OptionInfo(
                "f16",
                "LLM V cache type",
                gr.Dropdown,
                {"choices": [name for name, _ in _kv_types()]},
            ).info(
                "the value half of the cache. Some llama.cpp builds refuse a quantised V "
                "cache without flash attention; if llama-server will not start after "
                "changing this, put it back to f16"
            ),
        },
    )
)

VOICE_SECTION = ("model_chain_voice", "Voice Chat")
"""Its own section, not a corner of Model Chain's.

Voice is an install-level capability: somebody who has decided to try dictation
needs a stable place to provision two speech models, and burying that under a
heading about image model chaining would make it findable only by people who
already knew it was there. The two switches live with it because a settings page
that can install a feature and not turn it on is a settings page somebody has to
be told about twice.
"""


def _voice_row(builder=None):
    """One HTML settings row, as a component the host will actually draw.

    Forge's settings system stores *options*; it has no general way to host a
    Gradio button with a Python handler. ``OptionHTML`` is the supported escape
    hatch where a build has it, and a read-only HTML option is the fallback where
    it does not. Either way the buttons are wired by ``javascript/voice_chat.js``
    to the install route -- which is the design intent's own recommendation, and
    which keeps "download" from becoming a fake persistent boolean that a
    restored settings backup would re-trigger.

    Never fatal: a section that cannot draw its status row still draws its two
    switches, and Voice Chat is still installable from them plus the flyout.
    """
    try:
        markup = (builder or mc_voice_ui.settings_html)()
    except Exception:
        errors.report("Model Chain: could not build the Voice Chat settings row",
                      exc_info=True)
        return None
    html_option = getattr(shared, "OptionHTML", None)
    if html_option is not None:
        try:
            return html_option(markup)
        except Exception:
            errors.report("Model Chain: could not build the Voice Chat settings row",
                          exc_info=True)
    try:
        option = shared.OptionInfo(markup, "", gr.HTML)
        # Not a value anybody stores. Without this the markup would be written
        # into config.json as the "saved setting", and a settings backup would
        # restore last month's status line over this month's installation.
        option.do_not_save = True
        return option
    except Exception:
        errors.report("Model Chain: could not build the Voice Chat settings row",
                      exc_info=True)
        return None


_voice_status_row = _voice_row()
_voice_voices_row = _voice_row(mc_voice_ui.voices_html)

_VOICE_OPTIONS = {}
if _voice_status_row is not None:
    # First, because it is what the section is for: what is installed, and the
    # two buttons that install it. The switches under it are what to do with it.
    _VOICE_OPTIONS["model_chain_voice_status"] = _voice_status_row
if _voice_voices_row is not None:
    # Second: which voice speaks, auditioning it, and the optional cloning that
    # can add one. Below the installer because there is nothing to choose
    # between until the text-to-speech model is installed.
    _VOICE_OPTIONS["model_chain_voice_voices"] = _voice_voices_row

_VOICE_OPTIONS.update({
    mc_voice_engines.OPT_ENGINE: shared.OptionInfo(
        mc_voice_engines.DEFAULT_ENGINE,
        "Text-to-speech engine",
        gr.Radio,
        {"choices": list(mc_voice_engines.ENGINES)},
    ).info(
        "which engine speaks, for the whole WebUI. Kokoro is the built-in speaker bank; "
        "Sopro V2 is an optional streaming engine that makes a voice from a short recording "
        "and brings its own CPU runtime. Each engine keeps its own voices, default and "
        "per-character settings, so switching back restores what you had. Chosen with the "
        "cards in the row above rather than typed. Dictation is not part of this choice"
    ),
    mc_voice_pipeline.OPT_ENABLED: shared.OptionInfo(
        False,
        "Voice Pipeline",
    ).info(
        "off by default. On, generated speech is cleaned and its bandwidth restored "
        "before it is played, by two optional models that install separately and run on "
        "this PC's processor. It does not change the voice, the words or the speaking "
        "speed — it polishes what the speech engine already produced. The stages below "
        "run in a fixed order and only while PocketTTS is the selected engine"
    ),
    mc_voice_pipeline.OPT_DPDFNET: shared.OptionInfo(
        True,
        "Voice Pipeline: DPDFNet",
    ).info(
        "the first stage: takes noise and synthesis artefacts out of the generated "
        "speech, at whatever rate the engine produced it. Ticked by default so that "
        "turning the Voice Pipeline on gives you the intended chain in one gesture — the "
        "master switch above is what decides whether any of it runs"
    ),
    mc_voice_pipeline.OPT_LAVASR: shared.OptionInfo(
        True,
        "Voice Pipeline: LavaSR",
    ).info(
        "the second stage: restores the speech bandwidth a small model cannot generate "
        "and delivers 48 kHz. It runs after DPDFNet and never before it, because asking "
        "it to rebuild the top of a signal that still has hiss in it is asking it to "
        "rebuild the hiss. Ticked by default, like the stage above"
    ),
    mc_voice_pipeline.OPT_THREADS: shared.OptionInfo(
        mc_voice_pipeline.INTRAOP_THREADS,
        "Voice Pipeline: enhancement threads",
        gr.Number,
        {"precision": 0, "minimum": 1, "maximum": mc_voice_pipeline.MAX_INTRAOP_THREADS},
    ).info(
        "how many processor cores the enhancement stages may use. The default of two was "
        "chosen not to crowd the speech engine sharing those cores, and on a machine with "
        "many of them that default is what makes long replies break up: raise it while "
        "the \"Voice pipeline ran\" line in model_chain.log reports a real-time factor "
        "above 1.0, and stop when it is comfortably below. Takes effect on the next reply"
    ),
    mc_voice_device.OPT_DEVICE_DPDFNET: shared.OptionInfo(
        mc_voice_device.CPU,
        "Voice Pipeline: DPDFNet device",
        gr.Textbox,
    ).info(
        "which device this stage runs on. Set from the Voice Chat panel rather than "
        "typed here — the value is a card's identifier, not its name, because the number "
        "a card is given depends on who is counting. Nothing gives a chosen card back "
        "once the stage has loaded on it; a name this machine has no card for is ignored "
        "and the stage runs on the processor"
    ),
    mc_voice_device.OPT_DEVICE_LAVASR: shared.OptionInfo(
        mc_voice_device.CPU,
        "Voice Pipeline: LavaSR device",
        gr.Textbox,
    ).info(
        "which device this stage runs on. Set from the Voice Chat panel, like the stage "
        "above"
    ),
    mc_voice_state.OPT_AUTO_SEND: shared.OptionInfo(
        False,
        "Automatically send dictation",
    ).info(
        "off by default, so dictation fills the message box and you decide when to send. "
        "On, a successful transcription presses Send for you — through the same Send the "
        "keyboard uses, so attachments, threads and Stop all behave as they always do"
    ),
    mc_voice_state.OPT_AUTO_SPEAK: shared.OptionInfo(
        False,
        "Speak replies automatically",
    ).info(
        "off by default. On, each completed reply is read aloud by the local voice on this "
        "PC. A reply you stopped, or one that failed, is never spoken. The same switch is in "
        "Conversation's Voice menu"
    ),
    mc_voice_registry.OPT_VOICE: shared.OptionInfo(
        mc_voice_registry.DEFAULT_VOICE,
        "Default voice",
        gr.Textbox,
    ).info(
        "the stable id of the voice that reads replies aloud — chosen with Set as Default in "
        "the Voices list above rather than typed. A stable id rather than a speaker number "
        "on purpose: numbers move when a voice bank is rebuilt and this must not"
    ),
    mc_voice_registry.OPT_TEST_TEXT: shared.OptionInfo(
        mc_voice_registry.DEFAULT_TEST_TEXT,
        "Voice test text",
        gr.Textbox,
    ).info(
        "what Test says when you audition a voice. Editable here and in the Voices list"
    ),
    mc_voice_models.OPTIONS["stt"]: shared.OptionInfo(
        "",
        "Speech-to-text model",
        gr.Textbox,
    ).info(
        "which of the three transcription qualities is used — chosen with Use in the Speech "
        "to text row above rather than typed. Empty, or naming a model this build does not "
        "have, means the medium tier this extension ships with"
    ),
    # The four delivery numbers. Stored as options rather than in a file of
    # this feature's own for the reason the two switches are: Settings and the
    # Voice flyout are two views of one value, and a second store is a second
    # thing to keep in step. Sliders here as well as in the Voices row, because
    # somebody who has scrolled to the settings page should not have to find a
    # painted control to change a number they can see.
    mc_voice_profile.OPT_SPEED: shared.OptionInfo(
        mc_voice_profile.CONTROLS["speed"]["default"],
        "Voice speed",
        gr.Slider,
        {"minimum": mc_voice_profile.CONTROLS["speed"]["minimum"],
         "maximum": mc_voice_profile.CONTROLS["speed"]["maximum"],
         "step": mc_voice_profile.CONTROLS["speed"]["step"]},
    ).info(
        "Kokoro's own speaking rate for the default voice — the one control of the four "
        "that the model itself takes. Set it in the Delivery block above, which applies it "
        "the moment you let go of the slider; this row is the stored value. A character "
        "with a speed of its own overrides it, and one without follows it"
    ),
    mc_voice_profile.OPT_PITCH: shared.OptionInfo(
        mc_voice_profile.CONTROLS["pitch"]["default"],
        "Voice pitch (semitones)",
        gr.Slider,
        {"minimum": mc_voice_profile.CONTROLS["pitch"]["minimum"],
         "maximum": mc_voice_profile.CONTROLS["pitch"]["maximum"],
         "step": mc_voice_profile.CONTROLS["pitch"]["step"]},
    ).info(
        "shifts the whole voice, formants included, so it reads as a different-sized "
        "speaker. Kokoro has no pitch input of its own — Voice Chat resynthesises faster "
        "and reads the result back slower, which moves the formants with it. Set it in the "
        "Delivery block above"
    ),
    mc_voice_profile.OPT_GAIN: shared.OptionInfo(
        mc_voice_profile.CONTROLS["gain"]["default"],
        "Voice volume (dB)",
        gr.Slider,
        {"minimum": mc_voice_profile.CONTROLS["gain"]["minimum"],
         "maximum": mc_voice_profile.CONTROLS["gain"]["maximum"],
         "step": mc_voice_profile.CONTROLS["gain"]["step"]},
    ).info(
        "loudness relative to the model's own output, limited so a loud setting cannot clip "
        "into distortion. Set it in the Delivery block above"
    ),
    mc_voice_profile.OPT_PAUSE: shared.OptionInfo(
        mc_voice_profile.CONTROLS["pause"]["default"],
        "Pause between sentences (ms)",
        gr.Slider,
        {"minimum": mc_voice_profile.CONTROLS["pause"]["minimum"],
         "maximum": mc_voice_profile.CONTROLS["pause"]["maximum"],
         "step": mc_voice_profile.CONTROLS["pause"]["step"]},
    ).info(
        "extra silence after each sentence of a spoken reply, on top of the model's own. "
        "Only a reply that is streamed sentence by sentence has boundaries to put it at, so "
        "it does not change an audition. Set it in the Delivery block above"
    ),
    # Sopro's own settings. Named apart from Kokoro's throughout, which is what
    # makes I-3 -- "selecting or editing one engine cannot overwrite the other's
    # state" -- a property of the option names rather than of a check somebody
    # has to remember to write.
    # Sopro's twelve settings used to be listed here as well as in the panels
    # that are meant to be used, and that was two controls for one value. A host
    # option is a component on the settings page, "Apply settings" writes every
    # component on that page back into the store, and the page's copy is stamped
    # when the page is built -- so pressing Apply put back the default voice, the
    # delivery and the engine settings as they had been before anybody touched
    # the panel. They live in Sopro's own files now, next to the voices they
    # belong to, and the panel is the only place they are set.
    mc_voice_clone.OPT_ROOT: shared.OptionInfo(
        "",
        "Voice cloning folder",
        gr.Textbox,
    ).info(
        "where a prepared Storytime cloning bundle lives, if you have one. Empty looks in a "
        "cloning folder inside the voice data directory. Cloning is entirely optional — "
        "Voice Chat speaks without it, and a voice cloned once needs it never again"
    ),
    mc_voice_paths.OPT_ROOT: shared.OptionInfo(
        "",
        "Voice data directory",
        gr.Textbox,
    ).info(
        "where the CPU speech runtime and the two speech models are kept. Empty uses a "
        "model_chain_voice folder in your WebUI data directory. Separate from the LLM data "
        "directory on purpose: speech models have their own lifecycle and are not language "
        "models"
    ),
})

try:
    shared.options_templates.update(shared.options_section(VOICE_SECTION, _VOICE_OPTIONS))
except Exception:
    errors.report("Model Chain: could not register the Voice Chat settings section",
                  exc_info=True)

mc_progress.install()
"""Wrap the host's progress endpoint while rebinding it still takes effect.

``modules.progress.setup_progress_api`` registers the route long after
extensions are imported and resolves the function from the module globals when
it does, so this has to happen at import time and cannot wait for a UI callback.
Failure is not fatal -- see mc_progress.install().
"""


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


# --------------------------------------------------------------------------- #
# The Image Pipeline's context lines
# --------------------------------------------------------------------------- #
#
# Section 7: Stage 1 is Forge's, and this extension only reads it. Every number
# below comes out of a native control that remains the only place it can be
# changed -- there is no second width box here, and there never will be, because
# two controls holding one value is a bug with a delay on it.


def _short_checkpoint(name: str) -> str:
    """A checkpoint filename as something that fits on a summary line."""
    name = str(name or "").strip()
    if not name:
        return ""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    # The hash first. Forge appends it in square brackets *after* the extension,
    # so stripping the extension first finds nothing to strip and leaves
    # "krea2.safetensors" where "krea2" was wanted.
    if "[" in name:
        name = name.split("[", 1)[0]
    name = name.strip()
    for suffix in (".safetensors", ".ckpt", ".gguf", ".sft"):
        if name.casefold().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip() or str(name)


def _stage1_size(width, height, hires=False, hr_scale=2.0, hr_resize_x=0, hr_resize_y=0):
    """The pixels Stage 1 actually finishes with, Hires included.

    The one calculation on this panel that is worth more than it looks. The
    width and height sliders describe the *first pass*; with Hires on, what
    leaves Stage 1 -- and therefore what Stage 2 is handed -- is the upscaled
    image, and a panel that quoted the sliders would be describing a picture
    that never exists.
    """
    try:
        width, height = int(width or 0), int(height or 0)
    except (TypeError, ValueError):
        return 0, 0
    if width <= 0 or height <= 0:
        return 0, 0
    if not hires:
        return width, height
    try:
        return mc_arch.hires_target_size(
            width, height, float(hr_scale or 2.0),
            int(hr_resize_x or 0), int(hr_resize_y or 0))
    except (TypeError, ValueError, ZeroDivisionError):
        return width, height


def _handoff_note(width, height, hires=False, hr_scale=2.0, hr_resize_x=0,
                  hr_resize_y=0) -> str:
    """What crosses the edge into Stage 2, for Stage 2's own description.

    Said even when Stage 2 is off, because it is what Stage 2 *would* be handed
    and the number a user needs in order to decide whether to arm it.

    It had a row of its own on the connector between the two stages. That row
    is gone -- a pipeline of three cards should be three cards, and a fourth
    thing between two of them that is not a stage, cannot be opened and cannot
    be switched off is furniture.
    """
    width, height = _stage1_size(width, height, hires, hr_scale, hr_resize_x, hr_resize_y)
    return mc_pipeline_panel.handoff_note(width, height)


def _stage2_summary(enabled, target, denoise, multiplier, loaded="",
                    handoff="") -> str:
    """The Stage 2 card's description: what it is handed, and what it does.

    ``handoff`` is the Stage 1 size after any Hires pass -- the one number on
    this panel that nothing else on the page states. It leads, because it is
    true whether or not Stage 2 is armed, and because it is the number somebody
    reads in order to decide.

    Which is why the bypassed line drops the explanation rather than the number:
    the two together are wider than the header's second line, and a card header
    has exactly one line to say this in. With the size in front, "Bypassed" is
    the whole of what is left to say; it is only without one that the line has
    the room to spell out what being bypassed means.
    """
    lead = f"{handoff} · " if handoff else ""
    if not enabled:
        return f"{lead}Bypassed" if lead else "Bypassed — Stage 1 is final"

    parts = []
    name = _short_checkpoint(target)
    if name and target != _NO_MODEL:
        parts.append(name)
    else:
        parts.append("*no checkpoint chosen*")
    if loaded:
        parts.append(str(loaded))
    try:
        parts.append(f"denoise {float(denoise):.2f}".rstrip("0").rstrip("."))
    except (TypeError, ValueError):
        pass
    try:
        if float(multiplier or 1.0) > 1.0:
            parts.append(f"{float(multiplier):g}× size")
    except (TypeError, ValueError):
        pass
    return " · ".join(parts)


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
        # Native controls the Image Pipeline's Stage 1 row reads. Every one is
        # optional: a Forge build that renames one costs that clause of the
        # summary and nothing else, which is the only acceptable failure for a
        # panel that describes controls it does not own.
        self._checkpoint_component = None
        self._sampler_component = None
        self._scheduler_component = None
        self._steps_component = None
        self._cfg_component = None
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
        # The same one-shot for the Image Pipeline's context rows, which count
        # the same gallery.
        self._wire_pipeline_context = None
        # This generation's supplemental reference set, captured in process()
        # while the Stage 1 script arguments are still live.
        self._references = ()
        # The residency kind mc_memory.plan() predicted for this job's switch.
        self._transition = ""
        # Whether the host's own job counters were pre-sized for the whole chain.
        self._reserved = False
        # (entry timestamp, preload wait, Stage 1 preparation, bytes freed) for
        # the part of before_process that runs before the chain is known to be
        # armed. Consumed once, in process().
        self._preamble = ()

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
        # Read-only, for the Stage 1 context row. Section 2.5: these stay the
        # only place their values can be changed.
        "setting_sd_model_checkpoint": "_checkpoint_component",
        "txt2img_sampling": "_sampler_component",
        "txt2img_scheduler": "_scheduler_component",
        "txt2img_steps": "_steps_component",
        "txt2img_cfg_scale": "_cfg_component",
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
            self._wire_deferred_pipeline_context(component)

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

    def _wire_deferred_pipeline_context(self, gallery) -> None:
        """Finish the Image Pipeline's context rows, for the same reason.

        A one-shot on the same terms as the reference status above: dropped
        once used, so a rebuilt UI cannot leave two sets of handlers racing to
        write the same four lines.
        """
        wire, self._wire_pipeline_context = self._wire_pipeline_context, None
        if wire is None:
            return
        try:
            wire(gallery)
        except Exception:
            errors.report("Model Chain: failed to wire the Image Pipeline context",
                          exc_info=True)

    def ui(self, is_img2img):
        checkpoints, module_choices = _model_choices()
        samplers = _sampler_choices()
        schedulers = _scheduler_choices()
        styles = mc_styles.available_styles()

        pipeline = mc_pipeline_panel.host()

        # -- Stage 2's switch, on the pipeline row -------------------------- #
        #
        # The same boolean the accordion used to carry, in the place section 3.3
        # asks for it: on the collapsed row, so arming or bypassing Stage 2
        # never requires opening it. It is still the first control this script
        # returns and still the first field of a preset -- an InputAccordion is
        # a checkbox with a drawer attached, and the drawer moved.
        with pipeline.head("stage2"):
            enabled = mc_pipeline_panel.switch(
                elem_id=self.elem_id("enable"))

        with pipeline.body("stage2"):

            # -- 9.1 Presets ----------------------------------------------- #
            #
            # The first thing, because it is the one control that sets the
            # others. A preset here is the whole of Stage 2 -- checkpoint,
            # modules, sampling, seed policy, denoise, the lot -- so somebody
            # who has saved one makes a single choice and is finished, and the
            # sections below are where they go when they want to disagree with
            # part of it.
            #
            # It used to be the last accordion, under five others, with the
            # checkpoint chooser in this spot instead. That had the order of the
            # decisions backwards: the checkpoint is one of the things a preset
            # already decided.
            with gr.Group(elem_classes=mc_pipeline_panel.classes("essential")):
                with gr.Row():
                    preset = gr.Dropdown(
                        value=mc_presets.NONE,
                        label="Preset",
                        choices=mc_presets.choices(),
                        elem_id=self.elem_id("preset"),
                        info="applied the moment it is chosen",
                    )
                    preset_refresh = ToolButton(
                        value=refresh_symbol,
                        elem_id=self.elem_id("preset_refresh"),
                        tooltip="Presets: refresh",
                    )
                    # What the paragraph above this panel used to say, on the
                    # first control anybody reads.
                    mc_hint.control(
                        "Stage 2 finishes Stage 1 on the loaded checkpoint, then "
                        "re-encodes the result and refines it with a second "
                        "checkpoint. The handoff is in pixel space, so the two "
                        "models may use different architectures, VAEs and text "
                        "encoders. A preset carries every setting in this panel.",
                        label="Stage 2", elem_id=self.elem_id("intro"))

                preset_status = gr.Markdown("", elem_id=self.elem_id("preset_status"))
                preset_explain = gr.Markdown(
                    "", elem_id=self.elem_id("preset_explain"),
                    elem_classes=mc_pipeline_panel.classes("explain"))

                with mc_pipeline_panel.drawer("Save or delete a preset", elem_id=self.elem_id("section_presets")):
                    with gr.Row():
                        preset_name = gr.Textbox(
                            label="Preset name",
                            placeholder="name to save the current Stage 2 settings under",
                            elem_id=self.elem_id("preset_name"),
                            scale=3,
                        )
                        preset_save = gr.Button("Save", elem_id=self.elem_id("preset_save"), scale=1)
                        preset_delete = gr.Button("Delete", variant="stop",
                                                  elem_id=self.elem_id("preset_delete"),
                                                  scale=1)
                        # Whether Delete is armed. A gr.State and not a module
                        # variable: an arm is one person's half-finished gesture
                        # in one browser, and a flag on this process would be
                        # shared by every tab open on the server.
                        arm_preset_delete = gr.State(False)

                # The snapshot the dirty indicator compares against, and the
                # name it reports. Both are UI-only: a preset is applied to the
                # controls the moment it is chosen, exactly as before, and this
                # pair only remembers what those controls held at that moment.
                preset_baseline = gr.State("")
                preset_loaded = gr.State("")

                # The two dials somebody moves after choosing a preset, and the
                # only two: how far Stage 2 may take the picture, and how big it
                # comes back. Everything else below is a decision made once.
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

            # -- 9.2 Checkpoint & Model Components -------------------------- #
            #
            # A section like the others now. The checkpoint is not a thing
            # somebody picks fresh on every image -- it is part of what a preset
            # is -- and the modules and the residency status that describe the
            # same model belong beside it rather than five sections apart.
            with mc_pipeline_panel.drawer("Checkpoint & Model Components", elem_id=self.elem_id("section_modules")):
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

            # -- 9.3 Prompt & Styles --------------------------------------- #
            with mc_pipeline_panel.drawer("Prompt & Styles", elem_id=self.elem_id("section_prompt")):
                with gr.Row():
                    prompt_mode = gr.Radio(
                        choices=list(mc_infotext.PROMPT_MODES),
                        value="Inherit",
                        label="Stage 2 prompt",
                        elem_id=self.elem_id("prompt_mode"),
                    )
                    mc_hint.control(
                        "Flux-family models respond to natural-language phrasing "
                        "rather than comma-separated tags, so a Stage 2 prompt "
                        "often needs different wording than Stage 1. "
                        "<lora:name:weight> tags here are applied against the "
                        "Stage 2 model.",
                        label="the Stage 2 prompt",
                        elem_id=self.elem_id("prompt_hint"))
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

            # -- 9.4 Sampling ---------------------------------------------- #
            with mc_pipeline_panel.drawer("Sampling", elem_id=self.elem_id("section_sampling")):
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

            # -- 9.5 Edit & References ------------------------------------- #
            with mc_pipeline_panel.drawer("Edit & References", elem_id=self.elem_id("section_references")):
                edit_mode = gr.Radio(
                    choices=list(mc_arch.EDIT_MODES),
                    value=mc_arch.EDIT_AUTO,
                    label="Stage 2 edit mode",
                    elem_id=self.elem_id("edit_mode"),
                    info="Auto follows the global Settings toggle for the model",
                )
                edit_notice = gr.Markdown("", elem_id=self.elem_id("edit_notice"))

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

            # -- 9.6 Seed --------------------------------------------------- #
            with mc_pipeline_panel.drawer("Seed", elem_id=self.elem_id("section_seed")):
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


        # -- the memory contract ------------------------------------------- #
        #
        # Outside the Model Chain accordion, and deliberately. What it describes
        # is the memory policy for *this generation*, which exists whether or
        # not Stage 2 is armed: a plain Stage 1 press has a plan, a peak and a
        # persistent LLM allowance just as a long chain does, and a section that
        # only appeared alongside Stage 2 would suggest the policy did too.
        try:
            mc_plan_panel.build(self.elem_id)
        except Exception:
            errors.report("Model Chain: failed to build the generation memory section",
                          exc_info=True)

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

        def _preset_state(name, values):
            """The three things a load or a save leaves behind.

            The status line, the fingerprint later edits are compared against,
            and the name to report them under. Computed in one place because a
            baseline that disagrees with the name it was taken for is a dirty
            flag that reports on the wrong preset.
            """
            return (
                mc_profile_state.describe(name, False),
                "",
                mc_profile_state.snapshot(list(values)),
                name,
            )

        def on_preset_selected(name):
            skipped = [gr.skip()] * len(preset_controls)
            if not name or name == mc_presets.NONE:
                # Nothing named is loaded, so there is nothing to be modified
                # from. The controls keep whatever they hold -- deselecting a
                # preset is not a request to change any setting.
                return ["", "", "", "", *skipped]

            values = mc_presets.get(name)
            if values is None:
                return [
                    f'⚠️ Preset "{name}" no longer exists — refresh the list.',
                    "", gr.skip(), gr.skip(), *skipped,
                ]

            resolved = mc_presets.apply_defaults(values, preset_defaults)
            logger.info("Model Chain: applied preset %r", name)
            status, explain, baseline, loaded = _preset_state(
                name, [resolved[field] for field in mc_presets.FIELDS])
            return [
                status, explain, baseline, loaded,
                *[gr.update(value=resolved[field]) for field in mc_presets.FIELDS],
            ]

        preset.change(
            fn=on_preset_selected,
            inputs=[preset],
            outputs=[preset_status, preset_explain, preset_baseline, preset_loaded,
                     *preset_controls],
            show_progress=False,
        )

        # -- the modified indicator ---------------------------------------- #
        #
        # Section 8.2, and section 8.3 is why the sentence beside it is not
        # optional. A control that differs from the preset it was loaded from
        # says so on the panel, without a dialog and without being asked --
        # and says, in the same breath, that the edited value is the one the
        # next Generate will use. "Not saved" describes the file, never the
        # generation.
        #
        # `change` rather than `input` deliberately: it also fires when the
        # server writes a value, which is what makes applying a preset settle
        # back to a clean state instead of reporting the load as an edit.
        def on_preset_touched(loaded, baseline, *values):
            if not loaded or not baseline:
                return gr.skip(), gr.skip()
            modified = mc_profile_state.changed(list(values), baseline)
            return (mc_profile_state.describe(loaded, modified),
                    mc_profile_state.explain(modified))

        for control in preset_controls:
            control.change(
                fn=on_preset_touched,
                inputs=[preset_loaded, preset_baseline, *preset_controls],
                outputs=[preset_status, preset_explain],
                show_progress=False,
            )

        def on_preset_save(name, *values):
            try:
                saved = mc_presets.save(name, dict(zip(mc_presets.FIELDS, values)))
            except mc_presets.PresetError as exc:
                return f"⚠️ {exc}", gr.skip(), gr.skip(), gr.skip(), gr.skip()
            # Saving is what clears Modified, and it clears it by making the
            # stored copy match the screen -- not by changing anything on it.
            status, explain, baseline, loaded = _preset_state(name.strip(), values)
            return (
                status, explain, baseline, loaded,
                gr.update(choices=[mc_presets.NONE] + saved, value=name.strip()),
            )

        preset_save.click(
            fn=on_preset_save,
            inputs=[preset_name, *preset_controls],
            outputs=[preset_status, preset_explain, preset_baseline, preset_loaded,
                     preset],
            show_progress=False,
        )

        def on_preset_delete(name, armed):
            """Two presses, because deleting a preset removes a file.

            §3 of the pipeline intent asks for an explicit confirmation where
            the loss is irreversible. The confirmation is the button itself:
            the first press arms it and says which preset is about to go.
            """
            go, now, button = mc_pipeline_panel.confirmed(armed)
            if not go:
                return (now, button,
                        f'Press Delete again to remove the preset "{name}". '
                        "This cannot be undone.",
                        gr.skip(), gr.skip(), gr.skip(), gr.skip())
            try:
                remaining = mc_presets.delete(name)
            except mc_presets.PresetError as exc:
                return (now, button, f"⚠️ {exc}",
                        gr.skip(), gr.skip(), gr.skip(), gr.skip())
            return (
                now, button,
                f'Deleted preset "{name}".', "", "", "",
                gr.update(choices=[mc_presets.NONE] + remaining, value=mc_presets.NONE),
            )

        preset_delete.click(
            fn=on_preset_delete,
            inputs=[preset, arm_preset_delete],
            outputs=[arm_preset_delete, preset_delete,
                     preset_status, preset_explain, preset_baseline, preset_loaded,
                     preset],
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

        # -- the pipeline's context rows ------------------------------------ #
        #
        # Section 7. Stage 1 belongs to Forge and is read here, never written:
        # every value below arrives from a native control that stays the only
        # place it can be changed. The Output row and the handoff line are
        # derived from the same reads, so the three of them cannot disagree
        # about the size of the same picture.
        #
        # Optional throughout. A control this build of Forge does not expose is
        # simply absent from the inputs list, its clause is left out of the
        # sentence, and everything else still updates.
        _OBSERVED = ("checkpoint", "width", "height", "hires", "hr_scale",
                     "hr_resize_x", "hr_resize_y", "sampler", "scheduler",
                     "steps", "cfg")

        def _context_lines(read, is_on, target_name, denoise_value, multiplier,
                           loaded, gallery):
            """The two live lines, from one set of reads.

            There were four. The Stage 1 and Output rows they filled are gone
            from the panel -- the sliders and the picture were already saying
            those things -- and what is left is the pair nothing else on the
            page states: the size that crosses into Stage 2, and what Stage 2
            will do with it.
            """
            geometry = (read.get("width", 0), read.get("height", 0),
                        bool(read.get("hires", False)), read.get("hr_scale", 2.0),
                        read.get("hr_resize_x", 0), read.get("hr_resize_y", 0))
            return mc_pipeline_panel.card_summary(
                "stage2",
                _stage2_summary(bool(is_on), target_name, denoise_value,
                                multiplier, loaded,
                                handoff=_handoff_note(*geometry)))

        def wire_pipeline_context(stitch_gallery=None):
            """Connect the context rows, once it is settled what they can read.

            Deferred for the same reason the reference status is: ImageStitch
            sorts below Model Chain, so its gallery does not exist while this
            panel is being built, and a Gradio event captures its input list at
            registration.
            """
            outputs = [pipeline.summary("stage2")]
            if not all(outputs):
                return

            available = [
                (name, getattr(self, f"_{name}_component", None))
                for name in _OBSERVED
            ]
            present = [(name, component) for name, component in available
                       if component is not None]

            owned = [enabled, target, denoise, size_multiplier]
            inputs = [component for _, component in present] + owned + [preset_loaded]
            if stitch_gallery is not None:
                inputs.append(stitch_gallery)

            def refresh(*values):
                read = dict(zip([name for name, _ in present], values))
                rest = list(values[len(present):])
                is_on, target_name, denoise_value, multiplier = rest[:4]
                loaded = rest[4] if len(rest) > 4 else ""
                gallery = rest[5] if len(rest) > 5 else None
                return _context_lines(read, is_on, target_name, denoise_value,
                                      multiplier, loaded, gallery)

            triggers = [component for _, component in present] + owned
            if stitch_gallery is not None:
                triggers.append(stitch_gallery)
            for trigger in triggers:
                trigger.change(
                    fn=refresh,
                    inputs=inputs,
                    outputs=outputs,
                    show_progress=False,
                )

            # What the rows say before anybody has touched anything. Gradio
            # fires no event at page load, so the first render has to be written
            # into the components themselves -- which works because the config
            # the browser is built from is generated after every ui() has run.
            try:
                read = {name: getattr(component, "value", None)
                        for name, component in present}
                first = _context_lines(read, enabled.value, target.value,
                                       denoise.value, size_multiplier.value, "", None)
                for component, text in zip(outputs, first):
                    component.value = text
            except Exception:
                logger.debug("Model Chain: could not pre-render the pipeline context",
                             exc_info=True)

        if mc_references.stitch_is_installed():
            self._wire_pipeline_context = wire_pipeline_context
        else:
            wire_pipeline_context()

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

        What arrives as ``stage1_positive`` may already be the *inheritable*
        prompt rather than the generated one -- see the caller. That covers what
        a pattern cannot: a Krea literal command can carry any syntax at all,
        and the only reliable way to keep it out of Stage 2 is to inherit a
        prompt it was never written into.
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
        # The two spans below happen before this hook knows whether the chain is
        # armed, and both are real time the user spends looking at an empty
        # progress bar -- waiting on a preload that is still moving weights, and
        # freeing VRAM so Stage 1 loads in one piece. They are timed here and
        # handed to the estimator in process(), which is the first point that
        # has the whole job in front of it.
        entered = time.perf_counter()
        joined = entered
        prepared = entered
        freed = 0

        # Before anything is waited for or freed, because everything below is
        # sized against it. Published from here as well as from Creative Mode
        # so that a generation with no writer in it still has a plan: the panel
        # needs one to show, and a llama-server left running by LLM Studio needs
        # one to stay stable inside. Both scripts build it from ``p``, so the
        # answer does not depend on which of the two the host runs first.
        try:
            mc_plan.publish(mc_plan.build_for(p))
        except Exception:
            errors.report("Model Chain: could not build this generation's plan",
                          exc_info=True)

        # Which card this generation is about. Everything below is scoped to
        # it, because everything below is a statement about one physical GPU:
        # who has to wait for whom, and whose VRAM the image family is claiming.
        image_card = mc_broker.image_device_index()
        image_domain = mc_broker.image_execution_domain()

        # An LLM turn already in flight on *this card* gets to finish before
        # this generation touches it (section 12.3). This waits; it does not
        # take a lock, because there is no hook paired with this one that is
        # guaranteed to run afterwards and a lock left held here would block
        # LLM Studio until the WebUI restarted. Bounded, so a wedged runtime
        # delays a generation rather than preventing one.
        #
        # A conversation on another card, or on the processor, is not waited
        # for at all -- it is not using this GPU and never was, and the wait it
        # used to cause was a pure loss on both sides.
        # Everything this generation is going to load, loaded before it starts
        # rather than halfway through it. Off unless asked for, and a no-op
        # once the pipeline is already armed -- which after the first run of a
        # session it is, so this costs a measurement and nothing else.
        if mc_arm.mode() in (mc_arm.WARM_BEFORE, mc_arm.WARM_STARTUP):
            try:
                mc_arm.arm(p.width, p.height, reason="this generation")
            except Exception:
                errors.report("Model Chain: could not warm up before generating",
                              exc_info=True)

        mc_broker.await_idle(domain=image_domain)
        # Claim VRAM ownership for the image family, on the image card. The
        # number is zero on purpose: in Hybrid mode a generation starting is
        # not a reason to move anything, so this does nothing at all, and the
        # real demotion happens later, from make_vram_room, only if the pass
        # actually does not fit. In Exclusive mode it is the handover --
        # ownership there is a promise rather than an optimisation, so the LLM
        # leaves VRAM whether or not this generation would have needed the room
        # (sections 8 and 10). Ownership of *this* card: "the image family owns
        # the GPU" has never meant every GPU in the machine, and a sweep that
        # crossed to a second one would stop a language model to free memory
        # this generation cannot reach (section 9.2).
        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 0,
                               reason="an image generation started", margin=0,
                               card=image_card if image_card >= 0 else mc_broker.ANY_CARD)

        # Always reinstate a checkpoint we ourselves left swapped out, even when
        # the extension has since been disabled -- that is cleanup of our own
        # state, not extension behaviour.
        try:
            # Before anything else, and whether or not the extension is on: a
            # host left with no model but a loading hash that says otherwise
            # fails every generation from here on, and only this notices.
            mc_memory.ensure_model_loadable()
            # A preload started after the last generation may still be running.
            # Joining first means the worst case is the wait we would have had
            # anyway, never two threads moving weights at once.
            mc_memory.join_preload()
            joined = time.perf_counter()
            # Either branch means Stage 1 was swapped in from the cache; the
            # preload sized its VRAM budget from the *previous* generation, so
            # re-check it here against the size actually about to be sampled.
            swapped = mc_memory.reinstate_pending() or mc_memory.consume_preload()
            # Or whenever a language model is holding VRAM on this card. That
            # second reason is the only route Stage 1 has to the cross-workload
            # reclaim, and without it a plan with no Stage 2 had none: nothing
            # swaps, so nothing called make_vram_room, so llama-server was
            # never asked to give ground for a pass that did not fit. Forge's
            # own eviction cannot ask -- those bytes are in another process.
            # Silent and free when there is no language model on the card,
            # which is what keeps this off the console of somebody who does not
            # use that half of the extension.
            if swapped or mc_memory.llm_vram_on_the_image_card() > 0:
                freed = self._make_room_for_stage_1(p)
            if swapped or enabled:
                self._report_readiness()
            prepared = time.perf_counter()
        except Exception:
            errors.report("Model Chain: failed to reinstate the cached checkpoint", exc_info=True)

        self._armed = False
        self._dropped_networks = []
        self._transition = ""
        self._reserved = False
        self._preamble = (entered, joined - entered, prepared - joined, freed)
        # A plan left behind by a generation that never reached its end -- an
        # exception between here and postprocess -- would otherwise describe
        # this generation's bar with the last one's phases.
        mc_progress.abandon()
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
        # Kept for the estimator: the residency kind is what decides whether the
        # switch is a pointer swap, a copy back from system RAM or a disk read,
        # and those differ by an order of magnitude. Reading it from the same
        # call the console line uses means the prediction and the explanation
        # the user is given cannot disagree.
        self._transition = plan.kind
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
    def _make_room_for_stage_1(p) -> int:
        """Give Stage 1 a clean VRAM budget before it loads.

        Two situations reach here. In the first the previous generation ended
        with Stage 2's model in VRAM, and a warm swap does not evict it -- so
        Stage 1 would load into whatever is left. The host then makes room the
        hard way, partially unloading in small chunks while it loads, and that
        path is dramatically slower than a clean load: measured on a Krea 2 ->
        Flux.2 chain, the same ~8 GB UNet took 11.5s squeezed into 900 MB of
        spare VRAM against 0.8s with 7 GB spare.

        In the second a language model is holding VRAM on this card. That one
        is not about which of our models is loaded at all -- it is the only
        path Stage 1 has to the cross-workload reclaim, and the host cannot
        take that route itself because llama-server's bytes are in another
        process. "An image generation always outranks an idle LLM" is stated in
        the README as a rule rather than a setting, and on a plan that never
        swapped, nothing was in a position to enforce it.

        Stage 2 has had this since the pass-size fix; Stage 1 needs it for the
        same reasons. When everything genuinely fits, make_vram_room() checks
        first and does nothing.

        Freeing is all that happens here, and that boundary is deliberate.
        ``free_memory`` moves weights *out* to their offload device; loading
        them *in* rewrites and re-patches them, and doing that from here -- a
        hook, not the sampler -- proved able to leave the model in a torch
        state the following sampling step rejects outright with
        ``RuntimeError: Inference tensors do not track version counter``.
        The host's own lazy load, driven from inside the sampler, is in the
        right context by construction. Nothing is gained by beating it to it.

        Returns the bytes it freed, which is what the progress estimator learns
        its freeing rate from.
        """
        try:
            width, height = mc_arch.stage1_size(p)
            return int(
                mc_memory.make_vram_room(
                    shared.opts.sd_model_checkpoint,
                    mc_memory.current_modules(),
                    width,
                    height,
                    stage="Stage 1",
                )
                or 0
            )
        except Exception:
            errors.report("Model Chain: failed to free VRAM for Stage 1", exc_info=True)
            return 0

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

        # The first point with the whole job in view: Stage 1's geometry is
        # settled, Stage 2's target and multiplier are in hand, and no sampling
        # has started. Both of these size the job for the progress bar.
        self._reserve_job_counters(p, target, size_multiplier, int(steps))
        self._begin_progress(p, target, modules, size_multiplier, int(steps))

        styles = list(styles or [])
        if prompt_mode == "Inherit":
            styles = []

        # Per-image resolved prompts. create_infotext indexes list values by
        # image index, so a prompt that varies across the batch is recorded
        # accurately for each image rather than collapsing to image 0's.
        total = len(p.all_prompts or [p.prompt])
        # What a Stage 1 prompt-writing feature left for Stage 2, if one ran.
        # Two empty strings mean nobody rewrote this generation's prompt, and
        # ``all_prompts`` answers as it always has.
        #
        # One value for the whole batch rather than one per image, because that
        # is what it is: Creative Mode writes one prompt for the press, and the
        # per-image variation below it is the host's wildcard and style
        # expansion of *that* -- expansions of literal payloads Stage 2 is not
        # supposed to receive in the first place.
        inherit_positive, inherit_negative = mc_lora.stage1_inheritable(p)
        resolved_positive, resolved_negative = [], []
        for i in range(total):
            stage1_positive = inherit_positive or (
                p.all_prompts[i] if p.all_prompts else p.prompt)
            stage1_negative = inherit_negative or (
                p.all_negative_prompts[i] if p.all_negative_prompts else p.negative_prompt)
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

        It is also the only per-pass boundary the host offers, which makes it
        where the estimator is told a Stage 1 pass is starting -- including the
        hires second pass, which fires this hook again.
        """
        if not self._armed or self._in_stage_2:
            return

        mc_progress.note_pass()

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
            # No Stage 2 on this generation, so nothing after this point is
            # going to put the image model back on the card. The chained path
            # has done it since the preload existed -- it is the last thing
            # _run_stage_2 does -- and a single-stage plan never reached it,
            # which is why the model that had just finished sampling was found
            # cold again on the next click.
            if not self._in_stage_2:
                self._keep_stage_1_warm(p)
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
            mc_progress.end()
        except Exception:
            errors.report("Model Chain: Stage 2 failed; returning the unrefined Stage 1 images", exc_info=True)
            processed.comments += "\nModel Chain: Stage 2 failed — these images are unrefined Stage 1 output."
            # Dropped rather than closed: a job that failed part-way measures the
            # failure, not the work, and folding those spans into the store would
            # teach it that everything on this machine is faster than it is.
            mc_progress.abandon()
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
        mc_memory.observe_activation_peak(stage1_width, stage1_height, stage=mc_memory.STAGE_1,
                                          batch=getattr(p, "batch_size", 1))

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

        mc_progress.enter(mc_progress.PHASE_SWITCH)
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

        # Only when process() could not pre-size the counters. Adding to them
        # here is what made the bar fall backwards at the Stage 1 boundary --
        # Stage 1 has already driven job_no up to the old job_count by now -- so
        # it is a fallback for a host that did not accept the reservation rather
        # than the normal path.
        if not self._reserved:
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
        mc_progress.enter(mc_progress.PHASE_STAGE2_PREPARE)
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
        mc_progress.enter(mc_progress.PHASE_STAGE2)
        try:
            for index, image in enumerate(stage1_images):
                if state.interrupted or state.stopping_generation:
                    # Everything measured from here on would be measuring the
                    # interruption. The images already refined are still
                    # returned; the timings are not kept.
                    mc_progress.abandon()
                    break
                if state.skipped:
                    state.skipped = False

                state.job = f"Model Chain refine {index + 1}/{len(stage1_images)}"
                # The phase label the bar shows, which counts through the batch
                # without the percentage moving any differently for it.
                mc_progress.relabel(f"Stage 2 {index + 1}/{len(stage1_images)}")
                mc_progress.note_pass()

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
            # Everything from here is bookkeeping and delivery. The Stage 1
            # preload started at the end of it is deliberately *not* a phase:
            # it runs on a background thread after postprocess returns, by which
            # point the host has finished the task and removed the progress bar,
            # so "job complete" means the images are ready.
            mc_progress.enter(mc_progress.PHASE_FINALIZE)
            # Before the selection moves back, while shared.sd_model is still
            # Stage 2's. Completion or failure alike: the cache keeps this model
            # object, so anything left on it belongs to no job at all by the time
            # it is next used.
            self._clear_references(p)
            mc_memory.observe_activation_peak(
                stage_2_width, stage_2_height, stage=mc_memory.STAGE_2,
                batch=getattr(p, "batch_size", 1),
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
            mc_memory.preload_async(width, height, force=_warm_up_wanted())
        except Exception:
            errors.report("Model Chain: failed to start the Stage 1 preload", exc_info=True)

    @staticmethod
    def _keep_stage_1_warm(p) -> None:
        """Put the image model back on the card while the result is being looked at.

        A generation ends with its weights wherever the host's own eviction
        left them, which on a card sized for a checkpoint and a language model
        at once is "not all of them". Nothing moved them back until the next
        Generate click, so every generation paid the move -- and so did every
        LoRA, which is applied by walking the weights it patches and is
        therefore as slow as the bus when they are not on the card. A slow
        LoRA and a cold model were never two problems.

        Background and never waited on, for the same reason the chained
        preload is: this exists to overlap the time somebody spends looking at
        the images they just got. If it fails, is still running, or never
        starts, the next generation does exactly what it does today --
        before_process() joins the thread and then does the work itself.
        """
        try:
            width, height = mc_arch.stage1_size(p)
            mc_memory.preload_async(width, height, force=_warm_up_wanted())
        except Exception:
            errors.report("Model Chain: failed to start the Stage 1 warm-up", exc_info=True)

    @staticmethod
    def _extend_progress_total(extra_steps: int) -> None:
        """Add Stage 2's steps to the "Total progress" bar.

        Stage 1 sizes that bar for its own passes only -- and hires fix resizes
        it again for its second pass -- so Stage 2's steps would overflow it,
        making the bar wrap and re-render as though the work were repeating.
        Purely cosmetic, hence best-effort.

        Only reached when the counters were not pre-sized in process(); when
        they were, Stage 2's steps are already in the total.
        """
        if extra_steps <= 0:
            return
        try:
            bar = shared.total_tqdm._tqdm
            if bar is not None and bar.total:
                shared.total_tqdm.updateTotal(bar.total + extra_steps)
        except Exception:
            pass

    def _reserve_job_counters(self, p, target, size_multiplier, stage2_steps: int) -> None:
        """Size the host's own job and step counters for the whole chain, up front.

        Stage 2 used to be added to ``state.job_count`` in postprocess, by which
        point Stage 1 had already driven ``job_no`` to ``job_count`` and the bar
        to 100%; the addition then dropped it to a third of the way along. A
        batch of two refined after a 20-step Stage 1 went 100% -> 33% -> 66% ->
        100%. Setting the final figure before any sampling starts is what
        removes that, and it is what the bar falls back to when the progress
        endpoint could not be wrapped.

        The host would otherwise refine these numbers itself for a hires job,
        doubling ``job_count`` inside ``StableDiffusionProcessingTxt2Img.init``.
        Claiming that refinement here rather than compensating for it afterwards
        keeps one piece of arithmetic in one place -- and the console's step
        total is supplied along with it, so nothing is left half-adjusted.
        """
        self._reserved = False
        try:
            images = len(p.all_prompts or [p.prompt])
            passes = max(int(getattr(p, "n_iter", 1) or 1), 1)
            steps = max(int(getattr(p, "steps", 0) or 0), 0)
            hires = bool(getattr(p, "enable_hr", False))
            hr_steps = (int(getattr(p, "hr_second_pass_steps", 0) or 0) or steps) if hires else 0
            # The host's own special case: an upscale started from the gallery
            # has no first pass to count.
            first_steps = 0 if getattr(p, "txt2img_upscale", False) else steps

            state.job_count = passes * (2 if hires else 1) + images
            state.processing_has_refined_job_count = True
            shared.total_tqdm.updateTotal(
                passes * (first_steps + hr_steps) + images * max(stage2_steps, 0)
            )
            self._reserved = True
        except Exception:
            errors.report("Model Chain: could not pre-size the progress counters", exc_info=True)

    def _begin_progress(self, p, target, modules, size_multiplier, stage2_steps: int) -> None:
        """Hand the estimator a phase-by-phase plan of this generation.

        Everything it needs is known here and nowhere earlier: Stage 1's
        geometry is settled, the hires pass is decided, the Stage 2 target and
        multiplier are in hand, and no sampling has started.

        Best-effort throughout. A plan that cannot be built leaves the bar on
        the host's own arithmetic, which the counters above have just made
        monotonic anyway.
        """
        if not mc_progress.enabled():
            return

        try:
            passes = max(int(getattr(p, "n_iter", 1) or 1), 1)
            batch = max(int(getattr(p, "batch_size", 1) or 1), 1)
            steps = max(int(getattr(p, "steps", 0) or 0), 0)
            hires = bool(getattr(p, "enable_hr", False))
            hr_steps = (int(getattr(p, "hr_second_pass_steps", 0) or 0) or steps) if hires else 0

            base = _megapixels(getattr(p, "width", 0), getattr(p, "height", 0))
            stage1_width, stage1_height = mc_arch.stage1_size(p)
            upscaled = _megapixels(stage1_width, stage1_height)

            # Listed in the order they run rather than grouped by kind: with
            # hires on, each batch is a first pass followed by a second one,
            # and the two cost different amounts. Interleaving them is what
            # keeps the bar moving evenly through a multi-batch hires job.
            stage1_passes = []
            for _ in range(passes):
                stage1_passes.append((steps, base))
                if hires:
                    stage1_passes.append((hr_steps, upscaled))

            arch = mc_arch.detect_from_checkpoint_name(target)
            stage2_width, stage2_height = mc_arch.scaled_size(
                stage1_width, stage1_height, float(size_multiplier), arch.alignment
            )
            stage2_megapixels = _megapixels(stage2_width, stage2_height)

            # Every refine is its own batch-of-one pass, so a chain gets none of
            # Stage 1's batching gain and must not be predicted as though it
            # did. This is the whole reason batch size is part of the rate key.
            images = len(p.all_prompts or [p.prompt])
            stage2_passes = [(stage2_steps, stage2_megapixels)] * images

            job = mc_progress.build(
                stage1_arch=self._stage_1_arch_label(),
                stage1_passes=stage1_passes,
                batch_size=batch,
                stage2_arch=arch.label,
                stage2_passes=stage2_passes,
                transition=self._transition,
                move_gigabytes=mc_memory.file_size_bytes(target, modules) / _GB,
                free_gigabytes=self._expected_free_gigabytes(
                    target, modules, stage2_width, stage2_height
                ),
                target_label=os.path.splitext(os.path.basename(target))[0],
            )

            entered, waited, prepared, freed = self._preamble or (time.perf_counter(), 0.0, 0.0, 0)
            job.record(mc_progress.PHASE_JOIN, waited)
            job.record(mc_progress.PHASE_STAGE1_PREPARE, prepared, units=freed / _GB)
            mc_progress.begin(job, since=entered)
            mc_progress.enter(mc_progress.PHASE_STAGE1)
        except Exception:
            errors.report("Model Chain: could not plan this job's progress", exc_info=True)
            mc_progress.abandon()

    @staticmethod
    def _stage_1_arch_label() -> str:
        """The architecture rates are keyed on for Stage 1's sampling."""
        try:
            return mc_arch.detect_from_checkpoint_name(shared.opts.sd_model_checkpoint).label
        except Exception:
            return ""

    @staticmethod
    def _expected_free_gigabytes(target, modules, width: int, height: int) -> float:
        """How much VRAM the Stage 2 switch is likely to have to free.

        A prediction made while Model A is still resident, so it is the shortfall
        as it looks now rather than a reading. It only has to be the right order
        of magnitude: when the shortfall is nil the phase costs nothing, and
        make_vram_room() checks before acting for the same reason.
        """
        try:
            required = mc_memory.vram_required_bytes(target, modules, width, height)
            return max(required - mc_memory.free_vram_bytes(), 0) / _GB
        except Exception:
            return 0.0

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
    # First, and in a try of its own. Voice has nothing to do with VRAM, the
    # broker or the residency register, and invariant I-3 is not only about the
    # import graph: a speech worker that survived because releasing an image
    # model raised would be exactly the coupling this feature was designed
    # without.
    #
    # Every one of them, unconditionally, whichever engine was selected. "No
    # engine outlives the WebUI" is the requirement, and asking which one was
    # running is one more thing that can be wrong at the exact moment nothing
    # may be.
    #
    # One try *each*, and that is a correction rather than a style. All five
    # used to share a block, so a failure in the first left the other four
    # unstopped -- which is precisely the failure mode this door exists to
    # prevent, and section 34 says so in as many words: a failure in one must
    # not stop the cleanup of the others.
    for name, stop in (
            ("Kokoro", mc_voice_runtime.shutdown),
            ("Sopro", mc_voice_sopro_runtime.shutdown),
            # PocketTTS's ordinary Stop drains an abandoned unit; this is not
            # that. A lifecycle boundary ends the process rather than waiting
            # for quiescence, or a WebUI would refuse to close because a speech
            # engine was finishing a sentence nobody is listening to.
            ("PocketTTS", mc_voice_pocket_runtime.shutdown),
            # The cleanup engine, which has an idle timer of its own and must
            # not be trusted to have fired. A timer is a courtesy; this is the
            # requirement.
            ("the recording cleanup engine", mc_voice_cleanup_runtime.shutdown),
            # The Voice Pipeline, which ordinarily goes when PocketTTS goes and
            # is stopped here anyway. Its residency is coupled to an engine's;
            # its containment is not coupled to anything, and a door that
            # trusted the coupling would be a door that stays shut whenever the
            # coupling is the thing that broke (I-VP-20).
            ("the Voice Pipeline", mc_voice_pipeline_runtime.shutdown),
            # Separately owned, and stopped separately (section 82). A clone is
            # a long CPU job that has nothing to do with speech residency, and
            # its process tree must not outlive this one either.
            ("the Kokoro cloning tool", mc_voice_clone.shutdown)):
        try:
            stop()
        except Exception:
            errors.report(f"Model Chain: failed to stop {name}", exc_info=True)
    mc_memory.release_all()
    # The LLM lives in another process, so releasing our own references would
    # leave it running and holding VRAM after the extension has gone. It is
    # stopped first, and the residency register cleared after, so a reload
    # starts from an empty picture rather than from stale entries describing
    # models that are no longer anywhere.
    #
    # This is Forge asking an extension to tidy up, and it is not the exit
    # anybody performs: closing the window or killing the process never reaches
    # it. Those are covered by ``mc_llm_runtime.stop_on_exit``, which the first
    # start arms, and by the job object it describes.
    try:
        mc_llm_runtime.shutdown()
    except Exception:
        errors.report("Model Chain: failed to stop the LLM runtime", exc_info=True)
    mc_broker.clear()


try:
    from modules import script_callbacks

    script_callbacks.on_script_unloaded(_on_script_unloaded)
    script_callbacks.on_ui_tabs(mc_llm_studio.on_ui_tabs)
    # Both of these need the settings loaded, which is what on_app_started is:
    # the log file so that everything this extension says survives the terminal
    # window, and the route the Literal Prompt boxes report themselves over --
    # which is how "tag completion does not work in these boxes" gets answered
    # without asking somebody to open the developer tools. The file first, so
    # the report lands in it.
    script_callbacks.on_app_started(mc_arm.on_app_started)
    script_callbacks.on_app_started(mc_logfile.attach)
    script_callbacks.on_app_started(mc_literal_report.install)
    # The Voice Chat browser routes. Registered here rather than at import for
    # the reason every other route in this file is: there is no FastAPI app to
    # add anything to until the host has one.
    script_callbacks.on_app_started(mc_voice_api.install)
except Exception:
    errors.report("Model Chain: failed to register the extension callbacks", exc_info=True)
