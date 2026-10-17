"""The Model Chain theme: Forge's page dressed the way the user had it under
the Lobe theme, built here from scratch.

What it is. A Gradio *theme object* -- the host's own mechanism for a theme:
one object whose attributes become the CSS variables every Gradio control
reads, in a light set and a dark set -- installed in place of the host's choice
while the setting is on; a page script (``javascript/model_chain_theme.js``)
that lays out the header, the split view, the quick-settings drawer and the
drawn icons; and a stylesheet section keyed on the attribute that script
writes. The switches live in Settings → Model Chain: Theme and are read live
by the script, so turning one off puts the page back without a reload; only
the palette, being the host's theme object, waits for the next Reload UI.

What it is not. Nothing here comes from ``sd-webui-lobe-theme``. The user ran
that theme, liked a handful of its switches and declined its licence (handoff
43, ``docs/24-lobe-theme-host.md``), so this is written from a description of
the page -- a 64 px bar with the tabs in it, Generate above the gallery, the
prompt boxes at the top of the settings column, quick settings in a drawer,
drawn icons, a black page -- and from Forge's own DOM. The one thing read from
the Lobe folder is the user's own ``lobe_theme_config.json``: their data, read
once, so the switches here start where theirs were.

Why a theme object rather than a stylesheet of variables. A stylesheet that
redefines Gradio's variables fights Gradio's own sheet on order and
specificity, and has to be on the page before anything paints or the page
flashes. The theme object is what the host builds the page with
(``gr.Blocks(theme=shared.gradio_theme)``), it covers both modes, and it is
what the host's own *Gradio theme* setting produces -- so everything
downstream (Mini Paint NEO's WanGP hijack sampling the page's colours, the
quiet night mode reading ``--color-accent``) sees a theme like any other.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import types
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

SECTION = ("model_chain_theme", "Model Chain: Theme")
"""The Settings page section. A section of its own, next to Model Chain's."""

HIDDEN_SECTION = (None, "Model Chain: Theme (hidden)")
"""The host's idiom for a setting it stores but never draws: ``ui_settings``
skips any option whose section id is None."""

OPT_ENABLED = "model_chain_theme"
OPT_ACCENT = "model_chain_theme_accent"
OPT_FONT = "model_chain_theme_font"
OPT_TITLE = "model_chain_theme_title"
OPT_TAB_LABELS = "model_chain_theme_tab_labels"
OPT_SPLIT = "model_chain_theme_split"
OPT_SCROLL = "model_chain_theme_scroll"
OPT_QUICK = "model_chain_theme_quick"
OPT_QUICK_WIDTH = "model_chain_theme_quick_width"
OPT_PROMPTS = "model_chain_theme_prompts"
OPT_ICONS = "model_chain_theme_icons"
OPT_CONFIRM_UNLOAD = "model_chain_theme_confirm_unload"
OPT_NETWORKS = "model_chain_theme_networks"
OPT_NETWORKS_WIDTH = "model_chain_theme_networks_width"
OPT_NETWORKS_CARD = "model_chain_theme_networks_card"
OPT_NETWORKS_MODE = "model_chain_theme_networks_mode"
OPT_IMAGE_INFO = "model_chain_theme_image_info"
OPT_IMPORT = "model_chain_theme_import"
"""The record of the one-time import of the user's Lobe settings: hidden, so
the import runs once and what it did can be read back."""

OPT_HIDE_FOOTER = "model_chain_hide_footer"
"""Model Chain's own footer switch, by name: ``scripts/model_chain.py`` owns
it, and importing that module from here would be a cycle. A test holds the
two spellings together."""

QUICK_MODES = ("off", "collapsed", "open")
QUICK_DEFAULT = "collapsed"
QUICK_WIDTH_DEFAULT = 280
QUICK_WIDTH_MIN, QUICK_WIDTH_MAX = 200, 600
TAB_LABEL_MODES = ("as_forge", "spaced")
TAB_LABEL_DEFAULT = "as_forge"
TITLE_DEFAULT = "Forge"
NETWORKS_MODES = QUICK_MODES
NETWORKS_DEFAULT = "collapsed"
NETWORKS_WIDTH_DEFAULT = 340
NETWORKS_LAYOUTS = ("drawer", "column")
NETWORKS_LAYOUT_DEFAULT = "drawer"
NETWORKS_WIDTH_MIN, NETWORKS_WIDTH_MAX = 240, 800
NETWORKS_CARD_DEFAULT = 120
NETWORKS_CARD_MIN, NETWORKS_CARD_MAX = 60, 320
IMPORT_VERSION = 3
"""The import record's version. A record from an earlier build is read again
for the keys that build could not use yet (the extra-network drawer and the
generation-info panel came one round after the first import)."""

ACCENT_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
NIGHT_ACCENT = "#ffffff"
DAY_ACCENT = "#000000"
"""The accent with nothing chosen: white at night and black by day -- the
neutral primary the user's page had, and the reason the quiet night mode
exists."""

NAMED_ACCENTS = {
    "red": "#e5484d", "volcano": "#f0542d", "orange": "#f76b15", "gold": "#f5a524",
    "yellow": "#f5d90a", "lime": "#9bd60f", "green": "#30a46c", "cyan": "#12a5b8",
    "blue": "#2f6fed", "geekblue": "#4f63d2", "purple": "#8e4ec6", "magenta": "#d6409f",
    "kitchen": "#1f7aff",
}
"""Our own hex for the colour *names* the Lobe settings file can hold, so a
chosen primary colour survives the import as a colour and not as a word."""

FONT_STACK = ("ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif")
MONO_STACK = ("ui-monospace", "Cascadia Code", "JetBrains Mono", "Consolas", "Menlo", "monospace")

LIGHTBOX_ATTRIBUTES = ("sd_webui_modal_lightbox_toolbar_opacity",
                       "sd_webui_modal_lightbox_icon_opacity")
"""Two values Forge hangs on its theme object, which Gradio turns into CSS
variables like any other attribute and the lightbox reads. Carried over to
ours, or the lightbox's toolbar loses its opacity."""

GRADIO_THEME_VARIABLES = frozenset(
    name + suffix
    for name in (
        "body_background_fill", "body_text_color", "body_text_color_subdued",
        "background_fill_primary", "background_fill_secondary",
        "border_color_accent", "border_color_primary", "color_accent_soft",
        "link_text_color", "link_text_color_active", "link_text_color_hover",
        "link_text_color_visited", "shadow_drop", "shadow_drop_lg", "shadow_inset",
        "block_background_fill", "block_border_color", "block_info_text_color",
        "block_label_background_fill", "block_label_border_color", "block_label_text_color",
        "block_title_background_fill", "block_title_border_color", "block_title_text_color",
        "panel_background_fill", "panel_border_color", "chatbot_code_background_color",
        "checkbox_background_color", "checkbox_background_color_focus",
        "checkbox_background_color_hover", "checkbox_background_color_selected",
        "checkbox_border_color", "checkbox_border_color_focus", "checkbox_border_color_hover",
        "checkbox_border_color_selected", "checkbox_label_background_fill",
        "checkbox_label_background_fill_hover", "checkbox_label_background_fill_selected",
        "checkbox_label_border_color", "checkbox_label_border_color_hover",
        "checkbox_label_text_color", "checkbox_label_text_color_selected",
        "error_background_fill", "error_border_color", "error_text_color",
        "input_background_fill", "input_background_fill_focus", "input_background_fill_hover",
        "input_border_color", "input_border_color_focus", "input_border_color_hover",
        "input_placeholder_color", "loader_color", "slider_color", "stat_background_fill",
        "table_border_color", "table_even_background_fill", "table_odd_background_fill",
        "table_row_focus", "button_cancel_background_fill",
        "button_cancel_background_fill_hover", "button_cancel_border_color",
        "button_cancel_border_color_hover", "button_cancel_text_color",
        "button_cancel_text_color_hover", "button_primary_background_fill",
        "button_primary_background_fill_hover", "button_primary_border_color",
        "button_primary_border_color_hover", "button_primary_text_color",
        "button_primary_text_color_hover", "button_secondary_background_fill",
        "button_secondary_background_fill_hover", "button_secondary_border_color",
        "button_secondary_border_color_hover", "button_secondary_text_color",
        "button_secondary_text_color_hover",
    )
    for suffix in ("", "_dark")
) | frozenset((
    "color_accent", "block_border_width", "block_padding", "block_radius", "block_shadow",
    "block_label_radius", "block_label_right_radius", "container_radius", "form_gap_width",
    "layout_gap", "input_border_width", "input_padding", "input_radius", "input_shadow",
    "input_shadow_focus", "button_border_width", "button_large_padding", "button_large_radius",
    "button_small_padding", "button_small_radius", "button_shadow", "button_shadow_active",
    "button_shadow_hover", "checkbox_border_radius", "checkbox_border_width",
    "checkbox_label_border_width", "checkbox_shadow", "checkbox_label_shadow", "table_radius",
    "panel_border_width", "block_label_border_width", "block_title_border_width",
    "error_border_width", "shadow_spread",
))
"""The names Gradio 4's ``Base.set`` takes, as far as this theme uses them.
A value under a name outside this set never reaches ``set``: it is dropped
with a log line, because Gradio refuses an unknown keyword outright and a
theme that raises at ``on_before_ui`` is a page with no theme at all."""


# --------------------------------------------------------------------------- #
# The palette
# --------------------------------------------------------------------------- #


def valid_accent(value) -> str:
    """The accent setting as a hex colour, or "" for the automatic one."""
    text = str(value or "").strip()
    return text.lower() if ACCENT_PATTERN.match(text) else ""


def _luminance(hex_colour: str) -> float:
    def channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_text(hex_colour: str) -> str:
    """Black or white, whichever reads on the colour."""
    return "#000000" if _luminance(hex_colour) > 0.4 else "#ffffff"


def palette(accent: str = "") -> dict:
    """Every variable the theme sets, light and dark, from one accent choice.

    The dark set is the page as the user runs it: a black page, surfaces a
    step above it, lines a step above those, text at 90% white. The light set
    is the same page by day. Nothing in either is a gradient or a shadow,
    which is most of what makes the page look calm; the one shadow left is
    under a dropdown's list, which has to come off the page to be read.
    """
    chosen = valid_accent(accent)
    night = chosen or NIGHT_ACCENT
    day = chosen or DAY_ACCENT
    on_night = contrast_text(night)
    on_day = contrast_text(day)

    dark = {
        "body_background_fill": "#000000",
        "background_fill_primary": "#000000",
        "background_fill_secondary": "#141414",
        "block_background_fill": "#0d0d0d",
        "panel_background_fill": "#0d0d0d",
        "block_border_color": "#262626",
        "border_color_primary": "#262626",
        "panel_border_color": "#262626",
        "border_color_accent": "#3a3a3a",
        "block_label_background_fill": "#1a1a1a",
        "block_label_border_color": "#262626",
        "block_label_text_color": "#b3b3b3",
        "block_title_background_fill": "none",
        "block_title_border_color": "none",
        "block_title_text_color": "#e6e6e6",
        "block_info_text_color": "#8c8c8c",
        "body_text_color": "#e6e6e6",
        "body_text_color_subdued": "#8c8c8c",
        "link_text_color": "#9cc4ff",
        "link_text_color_hover": "#bcd7ff",
        "link_text_color_active": "#bcd7ff",
        "link_text_color_visited": "#9cc4ff",
        "input_background_fill": "#161616",
        "input_background_fill_hover": "#1d1d1d",
        "input_background_fill_focus": "#1d1d1d",
        "input_border_color": "#2b2b2b",
        "input_border_color_hover": "#3a3a3a",
        "input_border_color_focus": "#4a4a4a",
        "input_placeholder_color": "#5c5c5c",
        "button_secondary_background_fill": "#1a1a1a",
        "button_secondary_background_fill_hover": "#262626",
        "button_secondary_border_color": "#2b2b2b",
        "button_secondary_border_color_hover": "#3a3a3a",
        "button_secondary_text_color": "#d9d9d9",
        "button_secondary_text_color_hover": "#ffffff",
        "button_primary_background_fill": night,
        "button_primary_background_fill_hover": night,
        "button_primary_border_color": night,
        "button_primary_border_color_hover": night,
        "button_primary_text_color": on_night,
        "button_primary_text_color_hover": on_night,
        "button_cancel_background_fill": "#d93025",
        "button_cancel_background_fill_hover": "#b3261e",
        "button_cancel_border_color": "#d93025",
        "button_cancel_border_color_hover": "#b3261e",
        "button_cancel_text_color": "#ffffff",
        "button_cancel_text_color_hover": "#ffffff",
        "checkbox_background_color": "#161616",
        "checkbox_background_color_hover": "#1d1d1d",
        "checkbox_background_color_focus": "#1d1d1d",
        "checkbox_background_color_selected": night,
        "checkbox_border_color": "#3a3a3a",
        "checkbox_border_color_hover": "#4a4a4a",
        "checkbox_border_color_focus": "#4a4a4a",
        "checkbox_border_color_selected": night,
        "checkbox_label_background_fill": "#161616",
        "checkbox_label_background_fill_hover": "#1d1d1d",
        "checkbox_label_background_fill_selected": "#1d1d1d",
        "checkbox_label_border_color": "#2b2b2b",
        "checkbox_label_border_color_hover": "#3a3a3a",
        "checkbox_label_text_color": "#e6e6e6",
        "checkbox_label_text_color_selected": "#e6e6e6",
        "slider_color": night,
        "loader_color": night,
        "color_accent_soft": "#1f1f1f",
        "stat_background_fill": "#3a3a3a",
        "error_background_fill": "#2a1416",
        "error_border_color": "#7a2a2e",
        "error_text_color": "#ff8a8a",
        "table_border_color": "#262626",
        "table_even_background_fill": "#000000",
        "table_odd_background_fill": "#0d0d0d",
        "table_row_focus": "#1a1a1a",
        "chatbot_code_background_color": "#0d0d0d",
        "shadow_drop": "none",
        "shadow_drop_lg": "0 8px 24px rgba(0, 0, 0, 0.55)",
        "shadow_inset": "none",
    }
    light = {
        "body_background_fill": "#f4f4f4",
        "background_fill_primary": "#f4f4f4",
        "background_fill_secondary": "#e9e9e9",
        "block_background_fill": "#ffffff",
        "panel_background_fill": "#ffffff",
        "block_border_color": "#e2e2e2",
        "border_color_primary": "#e2e2e2",
        "panel_border_color": "#e2e2e2",
        "border_color_accent": "#c8c8c8",
        "block_label_background_fill": "#f0f0f0",
        "block_label_border_color": "#e2e2e2",
        "block_label_text_color": "#555555",
        "block_title_background_fill": "none",
        "block_title_border_color": "none",
        "block_title_text_color": "#1a1a1a",
        "block_info_text_color": "#6b6b6b",
        "body_text_color": "#1a1a1a",
        "body_text_color_subdued": "#6b6b6b",
        "link_text_color": "#1a5fd0",
        "link_text_color_hover": "#1448a3",
        "link_text_color_active": "#1448a3",
        "link_text_color_visited": "#1a5fd0",
        "input_background_fill": "#ffffff",
        "input_background_fill_hover": "#fafafa",
        "input_background_fill_focus": "#ffffff",
        "input_border_color": "#d9d9d9",
        "input_border_color_hover": "#bfbfbf",
        "input_border_color_focus": "#9e9e9e",
        "input_placeholder_color": "#9e9e9e",
        "button_secondary_background_fill": "#f0f0f0",
        "button_secondary_background_fill_hover": "#e4e4e4",
        "button_secondary_border_color": "#d9d9d9",
        "button_secondary_border_color_hover": "#bfbfbf",
        "button_secondary_text_color": "#333333",
        "button_secondary_text_color_hover": "#000000",
        "button_primary_background_fill": day,
        "button_primary_background_fill_hover": day,
        "button_primary_border_color": day,
        "button_primary_border_color_hover": day,
        "button_primary_text_color": on_day,
        "button_primary_text_color_hover": on_day,
        "button_cancel_background_fill": "#d93025",
        "button_cancel_background_fill_hover": "#b3261e",
        "button_cancel_border_color": "#d93025",
        "button_cancel_border_color_hover": "#b3261e",
        "button_cancel_text_color": "#ffffff",
        "button_cancel_text_color_hover": "#ffffff",
        "checkbox_background_color": "#ffffff",
        "checkbox_background_color_hover": "#fafafa",
        "checkbox_background_color_focus": "#fafafa",
        "checkbox_background_color_selected": day,
        "checkbox_border_color": "#bfbfbf",
        "checkbox_border_color_hover": "#9e9e9e",
        "checkbox_border_color_focus": "#9e9e9e",
        "checkbox_border_color_selected": day,
        "checkbox_label_background_fill": "#ffffff",
        "checkbox_label_background_fill_hover": "#fafafa",
        "checkbox_label_background_fill_selected": "#f0f0f0",
        "checkbox_label_border_color": "#d9d9d9",
        "checkbox_label_border_color_hover": "#bfbfbf",
        "checkbox_label_text_color": "#1a1a1a",
        "checkbox_label_text_color_selected": "#1a1a1a",
        "slider_color": day,
        "loader_color": day,
        "color_accent_soft": "#e9e9e9",
        "stat_background_fill": "#c8c8c8",
        "error_background_fill": "#fdecec",
        "error_border_color": "#f3b4b4",
        "error_text_color": "#b3261e",
        "table_border_color": "#e2e2e2",
        "table_even_background_fill": "#ffffff",
        "table_odd_background_fill": "#f7f7f7",
        "table_row_focus": "#efefef",
        "chatbot_code_background_color": "#f7f7f7",
        "shadow_drop": "none",
        "shadow_drop_lg": "0 8px 24px rgba(0, 0, 0, 0.18)",
        "shadow_inset": "none",
    }
    shape = {
        # Gradio's --color-accent has one value for both modes; the stylesheet
        # swaps in the day accent under the light theme from the two
        # attributes the theme object carries beside this (mc_theme_accent_*).
        "color_accent": night,
        "block_border_width": "1px",
        "block_padding": "16px",
        "block_radius": "8px",
        "block_shadow": "none",
        "block_label_radius": "6px",
        "block_label_right_radius": "6px",
        "block_label_border_width": "1px",
        "block_title_border_width": "0px",
        "container_radius": "8px",
        "panel_border_width": "1px",
        "form_gap_width": "1px",
        "layout_gap": "12px",
        "input_border_width": "1px",
        "input_padding": "10px",
        "input_radius": "6px",
        "input_shadow": "none",
        "input_shadow_focus": "none",
        "button_border_width": "1px",
        "button_large_padding": "10px 20px",
        "button_large_radius": "6px",
        "button_small_padding": "6px 12px",
        "button_small_radius": "6px",
        "button_shadow": "none",
        "button_shadow_active": "none",
        "button_shadow_hover": "none",
        "checkbox_border_radius": "4px",
        "checkbox_border_width": "1px",
        "checkbox_label_border_width": "1px",
        "checkbox_shadow": "none",
        "checkbox_label_shadow": "none",
        "error_border_width": "1px",
        "table_radius": "8px",
        "shadow_spread": "0px",
    }

    values = dict(shape)
    values.update(light)
    values.update({name + "_dark": value for name, value in dark.items()})
    return values


def accents(accent: str = "") -> tuple:
    """(day, night) accent for a setting value."""
    chosen = valid_accent(accent)
    return (chosen or DAY_ACCENT, chosen or NIGHT_ACCENT)


def build_theme(gr, *, accent: str = "", font: str = ""):
    """The theme object, from the settings: (theme, names dropped).

    ``gr`` is passed in rather than imported so the builder can be exercised
    against a stand-in; the host passes its Gradio. Values under names the
    installed Gradio does not know are dropped and reported, never raised.
    """
    names = [str(font).strip()] if str(font or "").strip() else []
    theme = gr.themes.Base(font=names + list(FONT_STACK), font_mono=list(MONO_STACK))
    values = palette(accent)
    supported = {name: value for name, value in values.items()
                 if name in GRADIO_THEME_VARIABLES and hasattr(theme, name)}
    dropped = tuple(sorted(set(values) - set(supported)))
    theme.set(**supported)
    day, night = accents(accent)
    # Plain attributes, which Gradio turns into --mc-theme-accent-day and
    # --mc-theme-accent-night, the way Forge's own lightbox values travel.
    theme.mc_theme_accent_day = day
    theme.mc_theme_accent_night = night
    theme._mc_theme = True
    return theme, dropped


def is_ours(theme) -> bool:
    return bool(getattr(theme, "_mc_theme", False))


# --------------------------------------------------------------------------- #
# The settings
# --------------------------------------------------------------------------- #


def settings(shared) -> dict:
    """The options of the Settings section, in the order they are drawn."""
    OptionInfo = shared.OptionInfo
    import gradio as gr  # noqa: F811 -- the host's, for the component classes

    return {
        OPT_ENABLED: OptionInfo(
            False,
            "Use the Model Chain theme",
        ).info(
            "a black page with calm surfaces, the tabs in a 64 px bar at the top, "
            "Generate above the gallery and the prompt boxes at the top of the settings "
            "column, quick settings in a drawer, drawn icons. The palette is a Gradio "
            "theme and takes a Reload UI; everything else here is read live. Stands "
            "down by itself while the Lobe theme extension is still enabled, so turn "
            "this on first and disable that one second"
        ),
        OPT_ACCENT: OptionInfo(
            "",
            "Accent colour",
        ).info(
            "a hex colour such as <code>#2f6fed</code>, used for the primary buttons, "
            "the sliders and the ticked boxes in both modes. Empty is white at night and "
            "black by day. Reload UI to apply"
        ),
        OPT_FONT: OptionInfo(
            "",
            "Typeface",
        ).info(
            "the name of a family on Google Fonts, such as <code>Inter</code>, loaded "
            "from there and put first; empty uses the system's own typeface. Reload UI "
            "to apply"
        ),
        OPT_TITLE: OptionInfo(
            TITLE_DEFAULT,
            "Title in the header",
        ),
        OPT_TAB_LABELS: OptionInfo(
            TAB_LABEL_DEFAULT,
            "Tab names",
            gr.Radio,
            {"choices": list(TAB_LABEL_MODES)},
        ).info(
            "<b>as_forge</b> keeps the names as Forge writes them · <b>spaced</b> "
            "puts a space before a capital or a digit and capitalises each word, so "
            "txt2img reads Txt 2 Img and LoRA stays LoRA"
        ),
        OPT_SPLIT: OptionInfo(
            True,
            "Generate above the gallery, prompts at the top of the settings column",
        ).info(
            "the split view: the prompt boxes, the tool row and the styles row lead "
            "the settings column, and Generate heads the gallery's panel on the right"
        ),
        OPT_SCROLL: OptionInfo(
            True,
            "One scroller for the page: the settings first, then the results",
        ).info(
            "a wheel, a finger, the scrollbar and the keyboard all move the page, "
            "and the page spends the movement on the settings column first, down to "
            "its end, with Generate, the gallery and its buttons locked where they "
            "rest; then on the results column, down to the generation details, with "
            "the settings held at their end. Never both at once. A Details button at "
            "the end of the gallery's button row jumps to the results' end and back. "
            "With the settings column in the Forge Assistant's panel the page scrolls "
            "the results alone and the column scrolls in the panel"
        ),
        OPT_QUICK: OptionInfo(
            QUICK_DEFAULT,
            "Quick settings",
            gr.Radio,
            {"choices": list(QUICK_MODES)},
        ).info(
            "<b>off</b> leaves the checkpoint row above the tabs as Forge draws it · "
            "<b>collapsed</b> moves it into a drawer on the left, opened from the "
            "header · <b>open</b> the same drawer, open at every load. The drawer "
            "remembers whether you left it open, per browser"
        ),
        OPT_QUICK_WIDTH: OptionInfo(
            QUICK_WIDTH_DEFAULT,
            "Quick settings drawer width (px)",
            gr.Slider,
            {"minimum": QUICK_WIDTH_MIN, "maximum": QUICK_WIDTH_MAX, "step": 10},
        ),
        OPT_PROMPTS: OptionInfo(
            True,
            "Resizable prompt boxes",
        ).info(
            "a grip on the corner of every prompt box, in a monospace face; off keeps "
            "Forge's own fixed, scrolling boxes"
        ),
        OPT_ICONS: OptionInfo(
            True,
            "Draw the tool buttons' icons",
        ).info(
            "the paste, clear, styles, refresh, seed, folder, save and send buttons "
            "get drawn icons in place of their emoji. Only a button whose whole text is "
            "one of those emoji is touched, never a span or a word"
        ),
        OPT_NETWORKS: OptionInfo(
            NETWORKS_DEFAULT,
            "Extra networks",
            gr.Radio,
            {"choices": list(NETWORKS_MODES)},
        ).info(
            "<b>off</b> leaves Forge's Textual Inversion, Checkpoints and LoRA pages as "
            "tabs beside Generation · <b>collapsed</b> moves those pages into a drawer on "
            "the right, opened from the header, with Generation always in view · "
            "<b>open</b> the same drawer, open at every load. The drawer remembers "
            "whether you left it open, per browser"
        ),
        OPT_NETWORKS_MODE: OptionInfo(
            NETWORKS_LAYOUT_DEFAULT,
            "Extra networks layout",
            gr.Radio,
            {"choices": list(NETWORKS_LAYOUTS)},
        ).info(
            "<b>drawer</b> lays the pages over the right edge of the page · "
            "<b>column</b> puts them beside the generation UI, which makes room for "
            "them. Either way the drawer's edge drags to a width, remembered per "
            "browser until the width setting changes"
        ),
        OPT_NETWORKS_WIDTH: OptionInfo(
            NETWORKS_WIDTH_DEFAULT,
            "Extra networks drawer width (px)",
            gr.Slider,
            {"minimum": NETWORKS_WIDTH_MIN, "maximum": NETWORKS_WIDTH_MAX, "step": 10},
        ),
        OPT_NETWORKS_CARD: OptionInfo(
            NETWORKS_CARD_DEFAULT,
            "Extra networks card size in the drawer (px)",
            gr.Slider,
            {"minimum": NETWORKS_CARD_MIN, "maximum": NETWORKS_CARD_MAX, "step": 2},
        ).info("the width of a card in the drawer; a card is half as tall again"),
        OPT_IMAGE_INFO: OptionInfo(
            True,
            "Generation info as a panel",
        ).info(
            "the parameters under the gallery drawn as a panel: the prompt and the "
            "negative prompt each in a block with a copy button, and every setting as a "
            "chip. Off leaves Forge's own text"
        ),
        OPT_CONFIRM_UNLOAD: OptionInfo(
            False,
            "Ask before leaving the page",
        ),
    }


def hidden_settings(shared) -> dict:
    return {
        OPT_IMPORT: shared.OptionInfo("", "The record of the Lobe settings import"),
    }


# --------------------------------------------------------------------------- #
# Importing the user's Lobe settings, once
# --------------------------------------------------------------------------- #

LOBE_CONFIG_NAME = "lobe_theme_config.json"

IGNORED_LOBE_KEYS = (
    # Features this theme does not build: Lobe's prompt editor, highlighter,
    # translations, animated background, web fonts from its CDN, its logo, and
    # the quick settings as a fixed column (the extra networks can be one:
    # `extraNetworkFixedMode` translates, since the third import).
    "i18n", "liteAnimation", "promptEditor", "enableHighlight", "neutralColor",
    "enableWebFont", "logoType", "logoCustomUrl", "sidebarFixedMode",
)


def find_lobe_config(extensions_dir) -> Path | None:
    """The newest ``lobe_theme_config.json`` one folder under ``extensions``.

    The file's own name is what identifies the folder as the Lobe theme's: a
    folder can be called anything, and the user's copy may be disabled, so
    neither the name nor the enabled list is asked."""
    root = Path(extensions_dir) if extensions_dir else None
    if root is None or not root.is_dir():
        return None
    found = []
    for child in root.iterdir():
        candidate = child / LOBE_CONFIG_NAME
        try:
            if candidate.is_file():
                found.append((candidate.stat().st_mtime, candidate))
        except OSError:
            continue
    if not found:
        return None
    found.sort(key=lambda pair: pair[0])
    return found[-1][1]


def translate(raw: dict) -> tuple:
    """(our option -> value, Lobe keys ignored) from a Lobe settings dict.

    Only keys that are there and of the expected shape translate; a key with
    the wrong shape is ignored like an unknown one. The stray keys such a file
    can hold -- the user's carries ``detail: "Not Found"``, a 404 body the
    theme once saved as a setting -- fall through here harmlessly.
    """
    applied = {}
    consumed = set()

    def boolean(key):
        value = raw.get(key)
        return value if isinstance(value, bool) else None

    if boolean("layoutSplitPreview") is not None:
        applied[OPT_SPLIT] = boolean("layoutSplitPreview")
        consumed.add("layoutSplitPreview")

    sidebar = boolean("enableSidebar")
    if sidebar is not None:
        consumed.add("enableSidebar")
        if not sidebar:
            applied[OPT_QUICK] = "off"
        else:
            expanded = boolean("sidebarExpand")
            applied[OPT_QUICK] = "open" if expanded else "collapsed"
        if boolean("sidebarExpand") is not None:
            consumed.add("sidebarExpand")
    width = raw.get("sidebarWidth")
    if isinstance(width, (int, float)) and not isinstance(width, bool):
        applied[OPT_QUICK_WIDTH] = int(min(QUICK_WIDTH_MAX, max(QUICK_WIDTH_MIN, width)))
        consumed.add("sidebarWidth")

    networks = boolean("enableExtraNetworkSidebar")
    if networks is not None:
        consumed.add("enableExtraNetworkSidebar")
        if not networks:
            applied[OPT_NETWORKS] = "off"
        else:
            applied[OPT_NETWORKS] = "open" if boolean("extraNetworkSidebarExpand") else "collapsed"
        if boolean("extraNetworkSidebarExpand") is not None:
            consumed.add("extraNetworkSidebarExpand")
    width = raw.get("extraNetworkSidebarWidth")
    if isinstance(width, (int, float)) and not isinstance(width, bool):
        applied[OPT_NETWORKS_WIDTH] = int(min(NETWORKS_WIDTH_MAX, max(NETWORKS_WIDTH_MIN, width)))
        consumed.add("extraNetworkSidebarWidth")
    card = raw.get("extraNetworkCardSize")
    if isinstance(card, (int, float)) and not isinstance(card, bool):
        applied[OPT_NETWORKS_CARD] = int(min(NETWORKS_CARD_MAX, max(NETWORKS_CARD_MIN, card)))
        consumed.add("extraNetworkCardSize")
    layout = raw.get("extraNetworkFixedMode")
    if layout in ("fixed", "float"):
        # Lobe's fixed sidebar is a column beside the generation UI; its
        # floating one a drawer over the page.
        applied[OPT_NETWORKS_MODE] = "column" if layout == "fixed" else "drawer"
        consumed.add("extraNetworkFixedMode")

    kind = raw.get("promptTextareaType")
    if kind in ("resizable", "scroll"):
        applied[OPT_PROMPTS] = kind == "resizable"
        consumed.add("promptTextareaType")

    for key, option in (("svgIcon", OPT_ICONS), ("confirmPageUnload", OPT_CONFIRM_UNLOAD),
                        ("layoutHideFooter", OPT_HIDE_FOOTER), ("enableImageInfo", OPT_IMAGE_INFO)):
        if boolean(key) is not None:
            applied[option] = boolean(key)
            consumed.add(key)

    title = raw.get("logoCustomTitle")
    if isinstance(title, str) and title.strip():
        applied[OPT_TITLE] = title.strip()
        consumed.add("logoCustomTitle")
    elif title == "":
        consumed.add("logoCustomTitle")  # no title asked for, which is not an oddity

    colour = raw.get("primaryColor")
    if isinstance(colour, str) and colour.strip():
        hex_colour = NAMED_ACCENTS.get(colour.strip().lower()) or valid_accent(colour)
        if hex_colour:
            applied[OPT_ACCENT] = hex_colour
            consumed.add("primaryColor")
    elif colour == "":
        consumed.add("primaryColor")

    ignored = [key for key in raw if key not in consumed]
    return applied, ignored


def import_lobe_settings(opts, extensions_dir, *, defaults: dict, save=None, now=None) -> dict | None:
    """Read the user's Lobe settings file once and start our switches there.

    Only a switch still at its default is written: one the user has already
    set here is theirs. The record goes into the hidden option, which is what
    makes this run once; nothing in the Lobe folder is changed. Returns the
    record, or None when there was nothing to do.
    """
    data = getattr(opts, "data", None)
    if not isinstance(data, dict):
        return None
    if data.get(OPT_IMPORT):
        return _import_again(opts, defaults=defaults, save=save, now=now)
    found = find_lobe_config(extensions_dir)
    if found is None:
        return None
    try:
        raw = json.loads(found.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not a JSON object")
    except Exception as error:
        logger.warning("Model Chain theme: could not read %s (%s); nothing imported",
                       found, error)
        return None

    applied, ignored = translate(raw)
    written = {}
    kept = []
    for name, value in applied.items():
        current = data.get(name)
        if current is not None and current != defaults.get(name):
            kept.append(name)
            continue
        opts.set(name, value)
        written[name] = value

    record = {
        "version": IMPORT_VERSION,
        "from": str(found),
        "modified": found.stat().st_mtime,
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now() if now else time.time())),
        "applied": written,
        "kept": kept,
        "ignored": ignored,
        "raw": raw,
    }
    opts.set(OPT_IMPORT, json.dumps(record, ensure_ascii=False))
    if save is not None:
        try:
            save()
        except Exception:
            logger.warning("Model Chain theme: the import could not be saved", exc_info=True)
    logger.info("Model Chain theme: imported %d setting(s) from %s (%s kept as you set them)",
                len(written), found, len(kept))
    return record


def _import_again(opts, *, defaults: dict, save=None, now=None) -> dict | None:
    """A record from an earlier build, read again for what that build could
    not use: the raw file is in the record, so no file is needed. Only keys
    neither applied nor kept the first time, and still at their defaults,
    are written; the record moves to this version and is saved once."""
    record = import_record(opts)
    if not record or int(record.get("version", 1) or 1) >= IMPORT_VERSION:
        return None
    raw = record.get("raw")
    if not isinstance(raw, dict):
        record["version"] = IMPORT_VERSION
        opts.set(OPT_IMPORT, json.dumps(record, ensure_ascii=False))
        return None
    applied, ignored = translate(raw)
    done = set(record.get("applied") or {}) | set(record.get("kept") or [])
    written = {}
    kept = list(record.get("kept") or [])
    data = opts.data
    for name, value in applied.items():
        if name in done:
            continue
        current = data.get(name)
        if current is not None and current != defaults.get(name):
            kept.append(name)
            continue
        opts.set(name, value)
        written[name] = value
    record["version"] = IMPORT_VERSION
    record["applied"] = dict(record.get("applied") or {}, **written)
    record["kept"] = kept
    record["ignored"] = ignored
    record["again"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(now() if now else time.time()))
    opts.set(OPT_IMPORT, json.dumps(record, ensure_ascii=False))
    if save is not None:
        try:
            save()
        except Exception:
            logger.warning("Model Chain theme: the import could not be saved", exc_info=True)
    if written:
        logger.info("Model Chain theme: imported %d more setting(s) from the earlier record",
                    len(written))
    return record


def import_record(opts) -> dict | None:
    """The record of the import, if one happened."""
    data = getattr(opts, "data", None)
    text = data.get(OPT_IMPORT) if isinstance(data, dict) else None
    if not text:
        return None
    try:
        record = json.loads(text)
    except Exception:
        return None
    return record if isinstance(record, dict) else None


# --------------------------------------------------------------------------- #
# Installing the theme object
# --------------------------------------------------------------------------- #

_state = types.SimpleNamespace(replaced=None, dropped=(), installed=False)


def extensions_dir() -> Path | None:
    try:
        from modules import paths_internal
        found = getattr(paths_internal, "extensions_dir", None)
        if found:
            return Path(found)
    except Exception:
        pass
    try:
        from modules import paths
        script_path = getattr(paths, "script_path", None)
        if script_path:
            return Path(script_path) / "extensions"
    except Exception:
        pass
    return None


def install(shared, gr) -> bool:
    """Put our theme object in the host's place, or the host's back; True if
    ours is in place afterwards.

    The host builds its theme object once at startup and again only when its
    own *Gradio theme* setting changes, so whatever object is there at
    ``on_before_ui`` is what the page is built with. Ours replaces it while
    the setting is on and remembers what it replaced; when the setting goes
    off, that object comes back -- without this, turning the theme off and
    reloading the UI would keep our palette until the next restart.
    """
    opts = shared.opts
    current = getattr(shared, "gradio_theme", None)
    enabled = bool(getattr(opts, OPT_ENABLED, False))
    if not enabled:
        if is_ours(current) and _state.replaced is not None:
            shared.gradio_theme = _state.replaced
        _state.installed = False
        return False

    theme, dropped = build_theme(gr, accent=getattr(opts, OPT_ACCENT, ""),
                                 font=getattr(opts, OPT_FONT, ""))
    if current is not None and not is_ours(current):
        _state.replaced = current
    for name in LIGHTBOX_ATTRIBUTES:
        source = current if current is not None else None
        if source is not None and hasattr(source, name):
            setattr(theme, name, getattr(source, name))
    shared.gradio_theme = theme
    _state.dropped = dropped
    _state.installed = True
    if dropped:
        logger.info("Model Chain theme: this Gradio does not know %s; left to its defaults",
                    ", ".join(dropped))
    return True


def on_before_ui():
    """The host's hook before it builds the page: import once, then install."""
    try:
        from modules import shared
    except Exception:
        return
    try:
        defaults = {name: info.default for name, info in
                    (getattr(shared, "options_templates", None) or {}).items()}
        import_lobe_settings(shared.opts, extensions_dir(), defaults=defaults,
                             save=lambda: shared.opts.save(shared.config_filename))
    except Exception:
        logger.warning("Model Chain theme: the Lobe settings import failed", exc_info=True)
    try:
        import gradio as gr
        install(shared, gr)
    except Exception:
        logger.warning("Model Chain theme: could not install the theme", exc_info=True)
