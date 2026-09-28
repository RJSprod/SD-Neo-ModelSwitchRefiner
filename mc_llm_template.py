"""The chat template llama-server is given: Gemma 4's own, with its thought marker kept.

Gemma 4's template is asymmetric by design. When thinking is off, every new
prompt ends with an empty thought block, and the model writes its reply after
it::

    <|turn>model\\n<|channel>thought\\n<channel|>   <- the reply is written here

and when that reply is written back into the history, the template's
``strip_thinking`` removes the block and the turn reads::

    <|turn>model\\n<the reply><turn|>

So each new prompt parts from the one llama.cpp has cached four tokens before
the last reply began, and everything after that point is read again: the model's
own last reply, every turn. From the user's llama-server log, one turn on the
Intel GPU read 394 new tokens after a reply of 331 -- nearly all of it the reply
it had just written, at about 80 tokens a second.

The fix is a copy of the model's own template in which a past model turn opens
with the same empty block its reply was generated after, when thinking is off.
It is passed to llama-server as ``--chat-template-file``, and the history then
matches the cache through the end of the last reply. llama-server picks the
chat format (``peg-gemma4``) from the template's markers, and the copy keeps
all of them, so replies are parsed exactly as before. Checked against
llama-server b10621 with Gemma 4's own template, from llama.cpp's
``models/ggml-vocab-gemma-4.gguf``: the new prompt's match with the cached one
moved from four tokens short of the last prompt to its whole length.

The copy is made only when the template has the asymmetry, recognised by all
three of its parts -- the turn opening, ``strip_thinking`` on model turns, and
the empty block in the generation prompt -- and the turn opening appears
exactly once. Any other template is passed over, and llama-server uses the
model's own as it always has.

Nothing in a single-turn request changes: the block is added only in front of
a *past* model turn, and a request that has none renders exactly as it did.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATE_FILE_FLAG = "--chat-template-file"
"""llama-server's template override. Taken with Jinja, which the server uses
by default."""

DIRNAME = "chat_templates"
"""Under the LLM data root: the copies, each named for its own content."""

OPT_KEEP_MARKER = "model_chain_llm_keep_thought_marker"
"""Whether Gemma's past replies keep the empty thought marker."""

TEMPLATE_KEY = "tokenizer.chat_template"

TURN_OPENING = "{{- '<|turn>' + role + '\\n' }}"
"""The template's opening of every turn, as it is written in its source."""

EMPTY_THOUGHT = "'<|channel>thought\\n<channel|>'"
"""The empty thought block as a Jinja string literal in the template's source."""

STRIPS_THINKING = "strip_thinking("
"""The macro that removes a thought block from a past model turn."""

KEPT = ("{%- if role == 'model' and not enable_thinking | default(false) -%}"
        "{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}")
"""What the copy adds after the turn opening: the block the generation prompt
ends with, under the same condition the generation prompt uses."""


def enabled() -> bool:
    import mc_broker

    value = mc_broker.option(OPT_KEEP_MARKER, True)
    if isinstance(value, str):
        return value.strip().casefold() not in ("false", "0", "off", "no")
    return bool(value)


def model_template(model) -> str:
    """The chat template the model file carries, or ``""``."""
    import mc_gguf

    described = mc_gguf.describe(model) if model is not None else None
    if described is None:
        return ""
    found = described.metadata.get(TEMPLATE_KEY)
    return found if isinstance(found, str) else ""


def patched(template: str) -> str | None:
    """``template`` with the thought marker kept on past model turns, or None.

    None for a template that is not Gemma 4's asymmetric one, and for one that
    already keeps the marker, so nothing is ever patched twice.
    """
    if not template or KEPT in template:
        return None
    if template.count(TURN_OPENING) != 1:
        return None
    if STRIPS_THINKING not in template or EMPTY_THOUGHT not in template:
        return None
    return template.replace(TURN_OPENING, TURN_OPENING + KEPT, 1)


def needs_it(model) -> bool:
    """Whether this model's template is one the fix applies to."""
    return patched(model_template(model)) is not None


def directory() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / DIRNAME


def template_file(model) -> Path | None:
    """The patched copy of ``model``'s template on disk, written if missing."""
    copy = patched(model_template(model))
    if copy is None:
        return None
    digest = hashlib.sha256(copy.encode("utf-8")).hexdigest()[:16]
    path = directory() / f"thought-marker-{digest}.jinja"
    try:
        if not path.is_file() or path.read_text(encoding="utf-8") != copy:
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = path.with_suffix(".tmp")
            staging.write_text(copy, encoding="utf-8")
            staging.replace(path)
    except OSError:
        logger.debug("Model Chain: could not write the chat template copy", exc_info=True)
        return None
    return path


def launch_flags(configuration) -> list[str]:
    """``--chat-template-file <copy>`` for a Gemma 4 template, when the build has the flag."""
    if not enabled():
        return []
    import mc_llm_runtime

    if not mc_llm_runtime.runtime_supports(TEMPLATE_FILE_FLAG, configuration):
        return []
    path = template_file(getattr(configuration, "model", None))
    if path is None:
        return []
    return [TEMPLATE_FILE_FLAG, str(path)]


__all__ = ["DIRNAME", "EMPTY_THOUGHT", "KEPT", "OPT_KEEP_MARKER", "TEMPLATE_FILE_FLAG",
           "TURN_OPENING", "enabled", "launch_flags", "model_template", "needs_it",
           "patched", "template_file"]
