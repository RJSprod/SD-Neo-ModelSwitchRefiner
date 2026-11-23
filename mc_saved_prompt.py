"""LoRA tags at the end of a prompt, one a line: saved, shown and recalled.

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

One move (:func:`prompt_master.krea.extra_networks.last`), wherever an image's
text is written or read back:

* **The text an image saves.** Forge's ``processing.create_infotext`` is
  wrapped once, at app start, and the wrapper rewrites the head of the text
  Forge built -- its ``Prompt:`` line, which may be several lines -- for every
  generation on both tabs. The prompt Forge generates from is not touched, so
  an image is made exactly as before. Every copy of the saved text comes
  through that one function: the PNG's text, the gallery's text under the
  picture, ``params.txt`` (Forge's paste with an empty prompt), a grid's text,
  the picture saved before hires fix, and Model Chain's Stage 2 text, which
  calls the function through Forge's module for this reason.
* **The text PNG Info shows.** The tab shows an image's text as the file holds
  it, so an image saved before the move -- or by Img2Img, or by another tool --
  showed its LoRAs wherever they were saved, the user's report once the saved
  and the recalled text were right (2026-11-23: "I went straight to png info to
  load the settings of a previous image. The lora all showed up at the front of
  the prompt!"). Forge's ``extras.run_pnginfo`` is wrapped before the UI is
  built, because the tab binds it as it is built, and its answer carries the
  moved text twice: in the parameters the tab shows and in the hidden box its
  Send to buttons read. The file is never written.
* **The text an image dropped on the prompt box puts there**, the whole saved
  text, which the box's paste button then reads (``images.image_data``, bound by
  the prompt row the same way).
* **The prompt a recalled image puts in the box.** ``on_infotext_pasted`` moves
  the tags of every recalled prompt, whatever made it, and Forge's runner of the
  paste hooks (``script_callbacks.infotext_pasted_callback``) is wrapped so the
  move runs once more after all of them: a hook another extension registers
  after this one, writing the prompt back from what it read, cannot put the
  tags in front again.

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
(:func:`extra_networks.last`) -- and so is a record another tool saved as JSON
where Forge keeps its text, and any text this cannot read. Nothing is computed
differently inside Forge's functions: each answer is built exactly as Forge
builds it and only the prompt in it is replaced, and only when it is where
Forge puts it. Any surprise leaves Forge's answer as it was.
"""

from __future__ import annotations

import html
import importlib
import inspect
import json
import logging
import re

logger = logging.getLogger("model_chain")

_WRAPPED = "_model_chain_saved_prompt"
"""Set on every wrapper here, so a second install finds it and does nothing."""

NEGATIVE = "Negative prompt:"
"""The line that ends the prompt in a saved text, as Forge's parse reads it."""

PARAMETERS = re.compile(r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
"""Forge Neo's ``re_param`` (``modules/infotext_utils.py``), for a host whose own
cannot be read: a saved text's last line is its parameters when it holds three
of these."""


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


# --------------------------------------------------------------------------- #
# Wrapping a host function, once
# --------------------------------------------------------------------------- #


def _wrap(module, name, make) -> bool:
    """Put ``make(original)`` in place of ``module.name``, once. True when a wrapper is there."""
    original = getattr(module, name, None)
    if not callable(original):
        return False
    if getattr(original, _WRAPPED, False):
        return True
    wrapper = make(original)
    setattr(wrapper, _WRAPPED, True)
    wrapper.__wrapped__ = original
    wrapper.__doc__ = original.__doc__
    setattr(module, name, wrapper)
    return True


def _unwrap(module, name) -> bool:
    """Put the host's own ``module.name`` back. True when there was a wrapper to take off."""
    current = getattr(module, name, None)
    if not getattr(current, _WRAPPED, False):
        return False
    setattr(module, name, current.__wrapped__)
    return True


def _host(name):
    """Forge's ``modules.<name>``, or ``None`` on a host without it."""
    try:
        return importlib.import_module(f"modules.{name}")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The text an image saves
# --------------------------------------------------------------------------- #


def hosts_function(module=None):
    """``(module, create_infotext)``, or ``(None, None)`` on a host without one."""
    if module is None:
        module = _host("processing")
        if module is None:
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

    def make(original):
        def create_infotext(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                return rewrite(result, prompt_of(signature, args, kwargs))
            except Exception:
                logger.debug("Model Chain: left a saved prompt as Forge wrote it", exc_info=True)
                return result
        return create_infotext

    return _wrap(module, "create_infotext", make)


def uninstall(module=None) -> bool:
    """Put Forge's own ``create_infotext`` back. True when there was a wrapper to take off."""
    module, _current = hosts_function(module)
    if module is None:
        return False
    return _unwrap(module, "create_infotext")


def on_app_started(_demo=None, _app=None) -> None:
    """Install the wrapper. Never raises; signature is ``script_callbacks.on_app_started``'s."""
    try:
        if not install():
            logger.warning("Model Chain: Forge has no create_infotext this build can wrap, "
                           "so saved prompts keep their LoRA tags where Forge put them")
    except Exception:
        logger.warning("Model Chain: could not wrap Forge's create_infotext", exc_info=True)


# --------------------------------------------------------------------------- #
# The text read back
# --------------------------------------------------------------------------- #


def _structured(text: str) -> bool:
    """Whether ``text`` is a JSON record rather than a prompt.

    Another tool can save one where Forge keeps its text (SwarmUI writes its
    parameters as JSON), and a tag lifted out of a JSON string would tear the
    record apart. A prompt that only starts with a brace -- a wildcard's
    ``{a|b}`` -- is not JSON and is moved as any other.
    """
    body = text.strip()
    if body[:1] not in ("{", "["):
        return False
    try:
        return isinstance(json.loads(body), (dict, list))
    except ValueError:
        return False


def recalled(prompt) -> str:
    """``prompt`` with its LoRA tags after the words, one a line.

    The text that goes back into the prompt box when an image is recalled,
    whatever made the image (2026-11-23: "regardless of the model"). Images
    saved before 2026-11-03 carry their tags in front -- the Krea hook's literal
    restore put them there and Forge saved that string -- and an image from
    Img2Img or another tool carries them where they were typed; all of them
    come back with the tags at the end. :func:`extra_networks.last` refuses
    whatever a move could change the meaning of (a line still holding ``[[``,
    a tag after a sign), a JSON record is left whole, and anything that is not
    text is handed back as it is.
    """
    if not isinstance(prompt, str) or _structured(prompt):
        return prompt
    from prompt_master.krea import extra_networks

    return extra_networks.last(prompt)


def _parameters():
    """Forge's own pattern for a parameter in a saved text's last line, else Forge Neo's copy."""
    host = _host("infotext_utils")
    pattern = getattr(host, "re_param", None)
    return pattern if hasattr(pattern, "findall") else PARAMETERS


def moved_text(text):
    """A whole saved text with its prompt's LoRA tags at the end, one a line.

    The prompt is found as Forge's ``parse_generation_parameters`` finds it:
    in the text stripped, the last line is the parameters when it holds three
    of them or more, and the prompt is every line before the first that begins
    ``Negative prompt:``. Only those lines are rewritten (:func:`recalled`); the
    negative prompt, the parameters and the whitespace around the text are kept
    byte for byte, so the text parses to the same parameters with the moved
    prompt. Anything that is not text, has no prompt or holds a prompt a move
    could change is handed back as it is.
    """
    if not isinstance(text, str):
        return text
    body = text.strip()
    if not body or _structured(body):
        return text
    lines = body.split("\n")
    end = len(lines)
    if len(_parameters().findall(lines[-1])) >= 3:
        end -= 1
    for number in range(end):
        if lines[number].lstrip().startswith(NEGATIVE):
            end = number
            break
    if end == 0:
        return text
    prompt = "\n".join(lines[:end])
    moved = recalled(prompt)
    if moved == prompt:
        return text
    start = len(text) - len(text.lstrip())
    return text[:start] + "\n".join([moved, *lines[end:]]) + text[start + len(body):]


def plaintext_to_html(text, classname=None):
    """Forge Neo's ``ui_common.plaintext_to_html``, for a host whose PNG Info names none."""
    content = "<br>\n".join(html.escape(line) for line in text.split("\n"))
    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"


def pnginfo_moved(answer, to_html=plaintext_to_html):
    """PNG Info's answer with the saved prompt's LoRA tags at the end.

    Forge's ``run_pnginfo`` answers ``("", text, shown)``: ``text`` is the
    image's saved text, which goes to the hidden box the tab's Send to buttons
    read, and ``shown`` is the HTML the tab shows -- the parameters first, each
    written with ``plaintext_to_html``. Both are given the moved text; ``shown``
    only where it holds the text as ``to_html`` writes it, so a tab Forge lays
    out some other way still shows what it showed.
    """
    if not isinstance(answer, tuple) or len(answer) != 3:
        return answer
    first, text, shown = answer
    moved = moved_text(text)
    if moved == text:
        return answer
    if isinstance(shown, str):
        old = to_html(text)
        if old and old in shown:
            shown = shown.replace(old, to_html(moved), 1)
    return first, moved, shown


def _pnginfo(module):
    def make(original):
        def run_pnginfo(*args, **kwargs):
            answer = original(*args, **kwargs)
            try:
                to_html = getattr(module, "plaintext_to_html", None)
                return pnginfo_moved(answer, to_html if callable(to_html) else plaintext_to_html)
            except Exception:
                logger.debug("Model Chain: PNG Info shows a saved prompt as the image holds it",
                             exc_info=True)
                return answer
        return run_pnginfo
    return make


def _dropped(_module):
    def make(original):
        def image_data(*args, **kwargs):
            answer = original(*args, **kwargs)
            try:
                if isinstance(answer, tuple) and answer and isinstance(answer[0], str):
                    moved = moved_text(answer[0])
                    if moved != answer[0]:
                        return (moved, *answer[1:])
            except Exception:
                logger.debug("Model Chain: a dropped image's text is put in the box as it is",
                             exc_info=True)
            return answer
        return image_data
    return make


def _pasted(_module):
    def make(original):
        def infotext_pasted_callback(*args, **kwargs):
            result = original(*args, **kwargs)
            params = args[1] if len(args) > 1 else kwargs.get("params")
            on_infotext_pasted(None, params)
            return result
        return infotext_pasted_callback
    return make


READERS = (
    ("extras", "run_pnginfo", _pnginfo, "PNG Info"),
    ("images", "image_data", _dropped, "an image dropped on the prompt box"),
    ("script_callbacks", "infotext_pasted_callback", _pasted, "Forge's paste"),
)
"""Where Forge reads an image's text back: its module, its function, the
wrapper, and what a log line calls it."""


def install_readers(**modules) -> list:
    """Wrap every reader in :data:`READERS`, once. The labels of those not wrapped.

    ``modules`` names a stand-in for a host module (``extras=...``), for tests;
    a module not named is Forge's own.
    """
    missing = []
    for host, name, wrapper, label in READERS:
        module = modules.get(host) or _host(host)
        try:
            wrapped = module is not None and _wrap(module, name, wrapper(module))
        except Exception:
            logger.debug("Model Chain: could not wrap %s", label, exc_info=True)
            wrapped = False
        if not wrapped:
            missing.append(label)
    return missing


def uninstall_readers(**modules) -> None:
    """Put each of Forge's readers back, so a reload wraps Forge's and not this one's."""
    for host, name, _wrapper, _label in READERS:
        module = modules.get(host) or _host(host)
        if module is not None:
            _unwrap(module, name)


def on_before_ui() -> None:
    """Wrap the readers before the UI binds them. Never raises.

    ``script_callbacks.on_before_ui``: Forge calls it before every build of the
    UI, the first and each after Reload UI, and PNG Info and the prompt row
    take their functions as they are built -- a wrapper put on after that would
    never be called by them.
    """
    try:
        missing = install_readers()
    except Exception:
        logger.warning("Model Chain: could not wrap what reads an image's saved text", exc_info=True)
        return
    if missing:
        logger.warning("Model Chain: this build of Forge has nothing to wrap for %s, so a saved "
                       "prompt read there keeps its LoRA tags where the image holds them",
                       ", ".join(missing))


def on_infotext_pasted(_infotext, params) -> None:
    """Put a recalled prompt's LoRA tags at its end, one a line. Never raises.

    ``script_callbacks.on_infotext_pasted``: Forge calls it with the parsed
    parameters before it fills a single field, from PNG Info's Send to buttons,
    from a gallery's Send to that reads the image's own text, and from the
    toprow's paste button -- so ``params["Prompt"]`` changed here is what the
    prompt box receives on every one of them. The wrapper on Forge's runner of
    these hooks calls it again after all of them (:data:`READERS`).
    """
    try:
        prompt = params.get("Prompt") if isinstance(params, dict) else None
        moved = recalled(prompt)
        if isinstance(moved, str) and moved != prompt:
            params["Prompt"] = moved
    except Exception:
        logger.debug("Model Chain: left a recalled prompt as it was saved", exc_info=True)
