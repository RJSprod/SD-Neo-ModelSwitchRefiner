"""The Voice Box tab: one HTML root, painted once, and nothing else of Gradio's.

Voice Box is a page a script owns (``javascript/voice_box.js``) rather than a
panel of Gradio components, for the reasons Mini Paint NEO learned the hard way
and ``docs/23-voice-box.md`` section 8 adopts: a component of another tab in an
outputs list is a silent, total failure; an equal string handed to a component
is no change at all; and everything the page shows is fetched, with a deadline,
from JSON routes on the page token rather than pushed through an event graph.

So Python's whole contribution is the root element -- its id, the token the
routes check, the routes' prefix and a boot flag -- and the script draws the
rest inside it. The token is the same one Voice Chat's routes check
(``mc_voice_api.session_token``): minted per WebUI process, so a tab left open
across a restart fails with "reload the page" rather than half-working.

This module imports only ``mc_voice_*`` modules, gradio and the standard
library, which keeps it inside invariant I-3 (``tests/test_voice_independence``):
a voice module never reaches the memory planner, the broker or the turns.
"""

from __future__ import annotations

import html
import logging

import gradio as gr

import mc_voice_api as api

logger = logging.getLogger(__name__)

TAB_LABEL = "Voice Box"
TAB_ID = "mc_voice_box"

ROOT_ID = "mc-voice-box"
"""The element the script boots on. Also the scope of every rule in the
``/* Voice Box */`` section of ``style.css``."""

ROOT_CLASS = "mc-voice-box"
ROOT_ELEM_ID = "mc_voice_box_root"
"""The Gradio component's own id, distinct from the root's: Gradio wraps the
HTML component's value in an element of that id, and the script looks for the
element *inside* it."""

PREFIX = "/model-chain/voice-box"
"""Where ``mc_voice_box_api`` mounts its routes. Carried on the root so the page
and the routes agree by construction rather than by two copies of a string."""

TOKEN_ATTRIBUTE = "data-mc-voice-key"
PREFIX_ATTRIBUTE = "data-mc-voice-box-prefix"
BOOT_ATTRIBUTE = "data-mc-voice-box-boot"


def root_markup(token: str | None = None) -> str:
    """The one element the page is built inside.

    Empty on purpose. The script clears whatever it finds under the root before
    drawing, so a static placeholder would be a second copy of the page's first
    words that had to be kept in step; an empty box is what a page whose script
    has not run looks like, and it is honest about it.
    """
    key = api.session_token() if token is None else token
    return (
        f'<div id="{ROOT_ID}" class="{ROOT_CLASS}" '
        f'{TOKEN_ATTRIBUTE}="{html.escape(key, quote=True)}" '
        f'{PREFIX_ATTRIBUTE}="{PREFIX}" '
        f'{BOOT_ATTRIBUTE}="1"></div>'
    )


def build() -> gr.Blocks:
    """The tab's Blocks: exactly one component, and no event binding.

    No outputs list exists here, so there is nothing for another tab's
    component to be put in; that is the point of the shape, not an accident of
    a first version.
    """
    with gr.Blocks(analytics_enabled=False) as blocks:
        gr.HTML(root_markup(), elem_id=ROOT_ELEM_ID)
    return blocks


def on_ui_tabs() -> list:
    """The ``on_ui_tabs`` callback. Never raises into the host.

    A tab that throws takes the whole WebUI's UI down with it, and Voice Box is
    never worth that: image generation and the rest of Model Chain must be
    unaffected by anything this tab does (design intent section 4F).
    """
    try:
        return [(build(), TAB_LABEL, TAB_ID)]
    except Exception:
        logger.warning("Model Chain: the Voice Box tab could not be built", exc_info=True)
        return []
