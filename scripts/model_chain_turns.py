"""The card's turn gate, on txt2img *and* img2img.

Model Chain's own script is txt2img-only -- Stage 1 is a txt2img generation by
definition -- and so is Creative Mode's. That was fine for everything those two
decide, and it is a hole in the one rule that is about the card rather than the
chain: a VibeVoice request goes next on its card, and every image job that has
not started waits for it (:mod:`mc_turns`). img2img and inpaint load the same
checkpoint onto the same card, and an inpaint started while VibeVoice renders on
it is two eighteen-gigabyte tenants on a twenty-four-gigabyte card.

So this is a third always-on script, shown on both tabs, with no controls and
three hooks. It draws nothing, adds nothing to presets or infotext, and costs a
dictionary check per generation until a speech guest has ever asked for a card.

On txt2img it is redundant on purpose. Model Chain's ``before_process`` calls the
gate first thing, because the host orders these hooks by script and a user can
reorder them; whichever call comes second in a job returns at once.

Since Cold LoRA (:mod:`mc_lora_ram`) it carries that rule's other hooks too, for
the same reason: a model with a LoRA baked in is the same model on img2img and
inpaint, and a prompt there can change the LoRA set just as well. Before the
host's LoRA pass the prompt is read against what is baked in; before sampling
the host's own record is checked; after the generation, in Cold, the merge is
baked. Each of the three is idempotent, so Model Chain's own calls on txt2img
and these can run in either order.
"""

from __future__ import annotations

import mc_lora_ram
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
        # Before the host reloads the model and parses the prompt's networks
        # (manage_model_and_prompt_cache runs right after this hook): a blob
        # the prompt would change is dropped here, so the host reads the
        # checkpoint and merges into fresh weights.
        try:
            mc_lora_ram.before_pass(p)
        except Exception:
            errors.report("Model Chain: the Cold LoRA check failed; generating anyway",
                          exc_info=True)

    def process_batch(self, p, *args, **kwargs):
        # After the host's LoRA pass and before sampling: the net under the
        # check above, for a rebuilt LoRA state over a blob.
        try:
            mc_lora_ram.before_sampling(p)
        except Exception:
            errors.report("Model Chain: the Cold LoRA check before sampling failed",
                          exc_info=True)

    def postprocess(self, p, processed, *args):
        try:
            mc_lora_ram.after_generation(p)
        except Exception:
            errors.report("Model Chain: baking the LoRA after the generation failed",
                          exc_info=True)
