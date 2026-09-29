"""The card's turn gate, on txt2img *and* img2img.

Model Chain's own script is txt2img-only -- Stage 1 is a txt2img generation by
definition -- and so is Creative Mode's. That was fine for everything those two
decide, and it is a hole in the one rule that is about the card rather than the
chain: a VibeVoice request goes next on its card, and every image job that has
not started waits for it (:mod:`mc_turns`). img2img and inpaint load the same
checkpoint onto the same card, and an inpaint started while VibeVoice renders on
it is two eighteen-gigabyte tenants on a twenty-four-gigabyte card.

So this is a third always-on script, shown on both tabs, with no controls and one
hook. It draws nothing, adds nothing to presets or infotext, and costs a
dictionary check per generation until a speech guest has ever asked for a card.

On txt2img it is redundant on purpose. Model Chain's ``before_process`` calls the
gate first thing, because the host orders these hooks by script and a user can
reorder them; whichever call comes second in a job returns at once.
"""

from __future__ import annotations

import mc_turns
from modules import errors, scripts


class ScriptModelChainTurns(scripts.Script):
    def title(self):
        return "Model Chain card turns"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args):
        try:
            mc_turns.image_gate(p)
        except Exception:
            errors.report("Model Chain: the card's turn check failed; generating anyway",
                          exc_info=True)
