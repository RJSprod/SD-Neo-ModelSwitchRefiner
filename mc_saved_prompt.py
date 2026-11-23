"""LoRA tags at the end of a prompt, one a line: saved and recalled.

On Txt2Img every extra-network tag in the prompt is lifted out before any
prompt writer can reword it, and put back *in front of* the finished prompt
(``prompt_master.krea.extra_networks.protect`` and ``literals.restore``). That
runs on every generation, Creative Mode on or off, and the ``Prompt:`` line
Forge saves into the image is that finished string -- so every image read back
in PNG Info had its LoRAs at the front, and the user asked for them at the end
(2026-11-03), on every model ("krea, moss, sdxl, etc"), and each on a line of
its own "so its very clean" (2026-11-23)::

    a woman in the rain, soft light
    <lora:realfilter_v2:0.85>
    <lora:contrast:-6>

Two places, one move (:func:`prompt_master.krea.extra_networks.last`):

* **The text an image saves.** Forge's ``processing.create_infotext`` is
  wrapped once, at app start, and the wrapper rewrites the head of the text
  Forge built -- its ``Prompt:`` line, which may be several lines -- for every
  generation on both tabs. The prompt Forge generates from is not touched, so
  an image is made exactly as before. Every copy of the saved text comes
  through that one function: the PNG's text, the gallery's text under the
  picture, ``params.txt`` (Forge's paste with an empty prompt), a grid's text,
  the picture saved before hires fix, and Model Chain's Stage 2 text, which
  calls the function through Forge's module for this reason.
* **The prompt a recalled image puts in the box.** ``on_infotext_pasted``
  moves the tags of every recalled prompt, whatever made it -- an image from
  before 2026-11-03 with its tags in front, an Img2Img image, another tool's.

What a paste of the moved text generates: on Txt2Img exactly what the old
order did, because the hook lifts the same tags in the same order wherever
they stand and puts them in front again before Forge reads the prompt
(``tests/test_saved_prompt.py`` holds that over a spread of prompts). On
Img2Img nothing lifts them, so Forge reads the box as it stands and takes the
tags out where they are: the words are the same, and what differs is only
what the tags' removal leaves behind -- a stray comma or space between the
words, which the move tidies away, and line breaks at the end rather than
spaces at the start. The user asked for the order regardless of that.

A prompt a move could change the meaning of is left as it was -- one still
holding a ``[[`` command, or with a tag right after a ``+`` or ``-``
(:func:`extra_networks.last`) -- and so is any text this cannot read. Nothing
is computed differently inside Forge's function: the text is built exactly as
Forge builds it (its ``Hires prompt`` comparison included, which reads the
prompt as Forge has it) and only the head of the result is replaced, and only
when the result starts with the prompt Forge put there. Any surprise leaves
Forge's text as it was.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger("model_chain")

_WRAPPED = "_model_chain_saved_prompt"
"""Set on the wrapper, so a second install finds it and does nothing."""


def prompt_of(signature, args, kwargs):
    """The prompt text Forge's ``create_infotext`` puts first, from its arguments.

    Worked out as Forge works it out (``modules/processing.py``): the main
    prompt for a grid, else the prompt at ``index``, which defaults to the
    position in the batch plus the batch's place in the job. ``None`` when the
    arguments do not bind, which leaves the text alone.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    values = bound.arguments
    p = values.get("p")
    if values.get("use_main_prompt"):
        return getattr(p, "main_prompt", None)
    index = values.get("index")
    if index is None:
        index = values.get("position_in_batch", 0) + values.get("iteration", 0) * p.batch_size
    return values["all_prompts"][index]


def rewrite(result, prompt_text):
    """``result`` with its prompt given the tags at the end, one a line.

    Forge returns ``f"{prompt}{negative}\\n{parameters}".strip()``, so the text
    starts with the prompt, less any leading whitespace (and less trailing
    whitespace when nothing follows it). Anything else is left as it was.
    """
    if not isinstance(result, str) or not isinstance(prompt_text, str):
        return result
    from prompt_master.krea import extra_networks

    moved = extra_networks.last(prompt_text)
    if moved == prompt_text:
        return result
    for head in (prompt_text.lstrip(), prompt_text.strip()):
        if head and result.startswith(head):
            return moved + result[len(head):]
    return result


def hosts_function(module=None):
    """``(module, create_infotext)``, or ``(None, None)`` on a host without one."""
    if module is None:
        try:
            from modules import processing as module  # type: ignore
        except Exception:
            return None, None
    original = getattr(module, "create_infotext", None)
    if not callable(original):
        return None, None
    return module, original


def install(module=None) -> bool:
    """Wrap Forge's ``processing.create_infotext``, once. True when the wrapper is in place."""
    module, original = hosts_function(module)
    if original is None:
        return False
    if getattr(original, _WRAPPED, False):
        return True
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return False

    def create_infotext(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            return rewrite(result, prompt_of(signature, args, kwargs))
        except Exception:
            logger.debug("Model Chain: left a saved prompt as Forge wrote it", exc_info=True)
            return result

    setattr(create_infotext, _WRAPPED, True)
    create_infotext.__wrapped__ = original
    create_infotext.__doc__ = original.__doc__
    module.create_infotext = create_infotext
    return True


def uninstall(module=None) -> bool:
    """Put Forge's own ``create_infotext`` back. True when there was a wrapper to take off."""
    module, current = hosts_function(module)
    if not getattr(current, _WRAPPED, False):
        return False
    module.create_infotext = current.__wrapped__
    return True


def on_app_started(_demo=None, _app=None) -> None:
    """Install the wrapper. Never raises; signature is ``script_callbacks.on_app_started``'s."""
    try:
        if not install():
            logger.warning("Model Chain: Forge has no create_infotext this build can wrap, "
                           "so saved prompts keep their LoRA tags where Forge put them")
    except Exception:
        logger.warning("Model Chain: could not wrap Forge's create_infotext", exc_info=True)


def recalled(prompt) -> str:
    """``prompt`` with its LoRA tags after the words, one a line.

    The text that goes back into the prompt box when an image is recalled,
    whatever made the image (2026-11-23: "regardless of the model"). Images
    saved before 2026-11-03 carry their tags in front -- the Krea hook's literal
    restore put them there and Forge saved that string -- and an image from
    Img2Img or another tool carries them where they were typed; all of them
    come back with the tags at the end. :func:`extra_networks.last` refuses
    whatever a move could change the meaning of (a line still holding ``[[``,
    a tag after a sign), and anything that is not text is handed back as it is.
    """
    if not isinstance(prompt, str):
        return prompt
    from prompt_master.krea import extra_networks

    return extra_networks.last(prompt)


def on_infotext_pasted(_infotext, params) -> None:
    """Put a recalled prompt's LoRA tags at its end, one a line. Never raises.

    ``script_callbacks.on_infotext_pasted``: Forge calls it with the parsed
    parameters before it fills a single field, from PNG Info's Send to buttons,
    from a gallery's Send to that reads the image's own text, and from the
    toprow's paste button -- so ``params["Prompt"]`` changed here is what the
    prompt box receives on every one of them.
    """
    try:
        prompt = params.get("Prompt") if isinstance(params, dict) else None
        moved = recalled(prompt)
        if isinstance(moved, str) and moved != prompt:
            params["Prompt"] = moved
    except Exception:
        logger.debug("Model Chain: left a recalled prompt as it was saved", exc_info=True)
