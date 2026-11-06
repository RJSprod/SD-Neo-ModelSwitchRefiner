"""LoRA tags at the end of an image's saved prompt.

On Txt2Img every extra-network tag in the prompt is lifted out before any
prompt writer can reword it, and put back *in front of* the finished prompt
(``prompt_master.krea.extra_networks.protect`` and ``literals.restore``). That
runs on every generation, Creative Mode on or off, and the ``Prompt:`` line
Forge saves into the image is that finished string -- so every image read back
in PNG Info had its LoRAs at the front, and the user asked for them at the end
(2026-11-03).

What this changes is the saved text and nothing else. Forge's
``processing.create_infotext`` is wrapped once, at app start, and the wrapper
moves the tags to the end of the ``Prompt:`` line of the text Forge built
(:func:`prompt_master.krea.extra_networks.last`). The prompt Forge generates
from is not touched, so an image is made exactly as before; and a paste of the
saved text gives Stage 1 the same prompt either way, because the tags are
lifted to the front again before Forge reads it.

Every copy of the saved text comes through that one function: the PNG's text,
the gallery's text under the picture, ``params.txt`` (Forge's paste with an
empty prompt), a grid's text, the picture saved before hires fix, and Model
Chain's Stage 2 text, which calls the function through Forge's module for this
reason.

Only a generation whose tags this extension lifted is rewritten: the Krea hook
marks its ``p`` (:func:`mark`) when a payload it placed holds a tag. Img2Img
has no such hook, so a tag there stays where it was typed -- moving it would
change what a paste of the text generates, because nothing there lifts it back.

Nothing is computed differently inside Forge's function: the text is built
exactly as Forge builds it (its ``Hires prompt`` comparison included, which
reads the prompt as Forge has it) and only the head of the result is
replaced, and only when the result starts with the prompt Forge put there. Any
surprise leaves Forge's text as it was.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger("model_chain")

MARK = "_model_chain_tags_lifted"
"""The attribute :func:`mark` sets on a processing object."""

_WRAPPED = "_model_chain_saved_prompt"
"""Set on the wrapper, so a second install finds it and does nothing."""


def mark(p) -> None:
    """Say that this generation's prompt had its tags lifted in front of it."""
    try:
        setattr(p, MARK, True)
    except Exception:
        logger.debug("Model Chain: could not mark the generation's saved prompt",
                     exc_info=True)


def lifted(p) -> bool:
    """Whether :func:`mark` was called on ``p``."""
    return bool(getattr(p, MARK, False))


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
    """``result`` with its first line's prompt given the tags at the end.

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
            p = args[0] if args else kwargs.get("p")
            if not lifted(p):
                return result
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
                           "so saved prompts keep their LoRA tags in front")
    except Exception:
        logger.warning("Model Chain: could not wrap Forge's create_infotext", exc_info=True)


LIFTED_KEYS = ("Krea Literal Command Count", "Krea Literal Syntax Version")
"""The image's own word that this extension lifted payloads out of its prompt.

``mc_infotext.LITERAL_COUNT`` and ``LITERAL_VERSION``, spelled out rather than
imported so this module stays importable without the infotext module's own
imports. Every generation the Krea hook lifted a LoRA tag in writes them (a
protected tag is a literal command), so they are what tells a recalled prompt
whose tags this extension put in front from one whose tags were typed there."""


def recalled(prompt, params) -> str:
    """``prompt`` with its LoRA tags at the end, when the image says they were lifted.

    The text that goes back into the prompt box when an image is recalled.
    Images saved before 2026-11-03 carry their LoRA tags in front -- the Krea
    hook's literal restore put them there and Forge saved that string -- and the
    user asked for them at the end in the prompt box as well as in the saved
    text (2026-11-06). Moving them changes nothing that is generated from the
    box on Txt2Img: :func:`prompt_master.krea.extra_networks.protect` lifts the
    same tags in the same order, wherever they stand, and the restore puts them
    in front again (``tests/test_saved_prompt.py`` holds that both saved lines
    paste to the same Stage 1 prompt).

    Only an image whose record says this extension lifted its tags is changed:
    a prompt without that record -- an Img2Img image, another tool's, one with
    no tag lifted -- has its tags where somebody typed them, and nothing here
    knows better. And :func:`extra_networks.last` refuses whatever a move could
    change the meaning of (a line still holding ``[[``, a tag after a sign).
    """
    if not isinstance(prompt, str) or not isinstance(params, dict):
        return prompt
    if not any(key in params for key in LIFTED_KEYS):
        return prompt
    from prompt_master.krea import extra_networks

    return extra_networks.last(prompt)


def on_infotext_pasted(_infotext, params) -> None:
    """Put a recalled prompt's LoRA tags at its end. Never raises.

    ``script_callbacks.on_infotext_pasted``: Forge calls it with the parsed
    parameters before it fills a single field, from PNG Info's Send to buttons,
    from a gallery's Send to that reads the image's own text, and from the
    toprow's paste button -- so ``params["Prompt"]`` changed here is what the
    prompt box receives on every one of them.
    """
    try:
        prompt = params.get("Prompt") if isinstance(params, dict) else None
        moved = recalled(prompt, params)
        if isinstance(moved, str) and moved != prompt:
            params["Prompt"] = moved
    except Exception:
        logger.debug("Model Chain: left a recalled prompt as it was saved", exc_info=True)
