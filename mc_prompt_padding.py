"""A batch's prompts of different lengths padded to the longest, as Forge did before 2026-10-30.

Before every sampling step Forge stacks the conditioning of every image in the
batch into one tensor (``modules/prompt_parser.py``: ``reconstruct_multicond_batch``
calls ``stack_conds``). Prompts of different lengths cannot be stacked as they
are. Until 2026-10-30 Forge Neo padded the shorter ones to the longest with
copies of their last token. Its commit ``49bc706`` ("batching") repeats every
prompt whole instead, up to the least common multiple of all the lengths, with
no limit. That is the idea ComfyUI and Forge's own cond/uncond batching
(``backend/sampling/condition.py``) use for CLIP's 77-token chunks, where the
multiple stays small, and both refuse a repeat past four times the shorter
prompt. Without that limit it was the user's out-of-memory error:

    Krea 2's text encoder gives every token 12 layers x 2560 float32 values
    (120 KiB), unpadded. Two wildcard picks a few tokens apart, 250 and 254
    tokens, become 31,750 tokens each: 7.3 GB of conditioning for a batch of
    two in place of 60 MB, built in system RAM and then moved onto the card
    beside a 12.6 GB model, which would then have to attend over 31,750 text
    tokens. Every generation with a wildcard and a batch above one ran a 24 GB
    card out of memory, and the same batch without the wildcard worked (the
    user confirmed it on 2026-10-30).

This module puts the old padding back, at the user's word ("I would like this
to work like the older builds"). Forge's ``stack_conds`` is wrapped once; a
call whose tensors differ in their first dimension alone -- their length -- is
answered with the old code's result, and every other call is the host's own:
tensors of one shape (every batch without a wildcard), or shapes that differ
elsewhere, which neither version stacks.

Tied to the host it was checked against by what the host does, not by its
build. The user's rule for an override of a host internal is that it is tied
to the host it was verified on; the batch-count override (``mc_batch_once``)
keeps it with a pin on the Forge build, which stands it down at every Forge
update until the switch is turned off and on again -- the user's log shows it
off from the first 2.31 start on. A pin here would bring the out-of-memory
error back with every update. So every start asks the host's function what it
makes of a 3-token and a 5-token tensor (two tiny tensors on the CPU): 15
tokens back is the uncapped repeat this was written against (15 is five times
3, past the limit of four), 5 is the padding it puts back. Only those two
answers are wrapped; anything else -- an error, another length -- is a host
this was not checked against, which keeps its own function, and the log warns
that the padding is off. The switch in Settings is read on every call.

The log says what the probe found when the wrapper goes in, and the first time
each set of lengths is padded in a session it says the lengths, what they were
padded to and, under the uncapped repeat, what the host would have built.
Token counts and sizes only: no prompt reaches the log.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("model_chain")

OPT_ENABLED = "model_chain_pad_batch_prompts"
ENABLED_DEFAULT = True

FORGE_CHANGE = "2026-10-30"
"""The day Forge Neo's ``stack_conds`` stopped padding (commit ``49bc706``)."""

REPEATS = "repeats"
"""The host repeats prompts of different lengths up to their least common multiple."""

PADS = "pads"
"""The host pads them to the longest, as Forge did before :data:`FORGE_CHANGE`."""

UNKNOWN = "unknown"
"""The host's answer could not be read: no torch, or an answer of another shape."""

_MARK = "_model_chain_pads_prompts"

_GB = 1024**3

_host: str = UNKNOWN
"""What the wrapped host function does with prompts of different lengths."""

_said: set = set()
"""The sets of lengths already said this session."""


def settings(shared) -> dict:
    """The option of the Settings page, registered by the main script in its section."""
    return {
        OPT_ENABLED: shared.OptionInfo(
            ENABLED_DEFAULT,
            "Pad a batch's prompts to the longest, as Forge did before October 30",
        ).info(
            "When the images of one batch have prompts of different lengths, as "
            "wildcards give them, Forge now repeats each prompt up to a common multiple "
            "of the lengths, which with Krea 2's text encoder runs the card out of "
            "memory. On pads the shorter prompts to the longest instead, as Forge Neo "
            "did until 2026-10-30"
        ),
    }


def enabled() -> bool:
    import mc_memory

    return bool(mc_memory.option(OPT_ENABLED, ENABLED_DEFAULT))


def lengths_differ(tensors) -> bool:
    """True when ``tensors`` share every dimension but their first, and that one differs."""
    try:
        shapes = [tuple(int(size) for size in x.shape) for x in tensors]
    except Exception:
        return False
    if len(shapes) < 2 or any(not shape or shape[0] < 1 for shape in shapes):
        return False
    if any(shape[1:] != shapes[0][1:] for shape in shapes):
        return False
    return len({shape[0] for shape in shapes}) > 1


def padded(tensors):
    """Forge Neo's ``stack_conds`` before 2026-10-30, for tensors of different lengths.

    Line for line what ``modules/prompt_parser.py`` did: each tensor shorter
    than the longest is followed by copies of its last token until it is as
    long, and the tensors are stacked.
    """
    import torch

    token_count = max(x.shape[0] for x in tensors)
    new_tensors = []
    for x in tensors:
        if x.shape[0] == token_count:
            new_tensors.append(x)
        else:
            diff = token_count - x.shape[0]
            last_vector = x[-1:]
            repeats = [diff] + [1] * (x.dim() - 1)
            new_tensors.append(torch.cat([x, last_vector.repeat(repeats)], dim=0))
    return torch.stack(new_tensors)


def probe(original) -> str:
    """What ``original`` makes of a 3-token and a 5-token tensor: :data:`REPEATS`,
    :data:`PADS` or :data:`UNKNOWN`. Never raises."""
    try:
        import torch

        answer = original([torch.zeros(3, 1), torch.ones(5, 1)])
        length = int(answer.shape[1])
    except Exception:
        return UNKNOWN
    if length == 15:
        return REPEATS
    if length == 5:
        return PADS
    return UNKNOWN


def hosts_function(module=None):
    """``(module, stack_conds)`` of Forge's prompt parser, or ``(None, None)``."""
    if module is None:
        try:
            from modules import prompt_parser as module
        except Exception:
            return None, None
    original = getattr(module, "stack_conds", None)
    if not callable(original):
        return None, None
    return module, original


def install(module=None) -> bool:
    """Wrap Forge's ``prompt_parser.stack_conds``, once. True when the wrapper is in place.

    False on a host without the function, and on one whose answer to the probe
    is neither the repeat nor the padding: that host keeps its own function.
    """
    global _host

    module, original = hosts_function(module)
    if original is None:
        return False
    if getattr(original, _MARK, False):
        return True
    _host = probe(original)
    if _host == UNKNOWN:
        return False

    def stack_conds(tensors):
        try:
            take = lengths_differ(tensors) and enabled()
        except Exception:
            take = False
        if not take:
            return original(tensors)
        result = padded(tensors)
        _say(tensors)
        return result

    setattr(stack_conds, _MARK, True)
    stack_conds.__wrapped__ = original
    stack_conds.__doc__ = original.__doc__
    module.stack_conds = stack_conds
    return True


def uninstall(module=None) -> bool:
    """Put the host's own ``stack_conds`` back. True when there was a wrapper to take off."""
    module, current = hosts_function(module)
    if not getattr(current, _MARK, False):
        return False
    module.stack_conds = current.__wrapped__
    return True


def on_app_started(_demo=None, _app=None) -> None:
    """Install the wrapper and say what the host does. Never raises.

    Signature is ``script_callbacks.on_app_started``'s.
    """
    try:
        module, original = hosts_function()
        if original is None:
            logger.debug("Model Chain: Forge has no prompt_parser.stack_conds to wrap")
            return
        if not install(module):
            logger.warning(
                "Model Chain: Forge's prompt stacking answered a batch of a 3- and a 5-token "
                "prompt with neither the padding nor the repeat this extension was checked "
                "against, so the padding of a batch's prompts is off on this Forge build")
            return
        if _host != REPEATS:
            return
        if enabled():
            logger.info(
                "Model Chain: Forge repeats a batch's prompts of different lengths up to a "
                "common multiple of their lengths; they are padded to the longest instead, "
                "as Forge did before %s", FORGE_CHANGE)
        else:
            logger.info(
                "Model Chain: Forge repeats a batch's prompts of different lengths up to a "
                "common multiple of their lengths, and the padding switch is off, so that "
                "stands; with an LLM text encoder it can run the card out of memory")
    except Exception:
        logger.debug("Model Chain: could not install the prompt padding", exc_info=True)


def _say(tensors) -> None:
    """One line the first time this session a set of lengths is padded."""
    try:
        lengths = [int(x.shape[0]) for x in tensors]
        key = tuple(sorted(set(lengths)))
        if key in _said:
            return
        _said.add(key)
        line = (f"Model Chain: a batch's prompts were {_span(key)} tokens long and were "
                f"padded to {max(lengths):,}, as Forge did before {FORGE_CHANGE}")
        if _host == REPEATS:
            common = math.lcm(*key)
            line += f"; Forge's own stacking would have repeated each to {common:,} tokens"
            size = len(tensors) * common * _token_bytes(tensors[0])
            if size:
                line += f", {size / _GB:.1f} GB for the batch"
        logger.info(line)
    except Exception:
        logger.debug("Model Chain: could not describe the padded prompts", exc_info=True)


def _span(key) -> str:
    if len(key) == 2:
        return f"{key[0]:,} and {key[1]:,}"
    return f"{key[0]:,} to {key[-1]:,}"


def _token_bytes(tensor) -> int:
    """The bytes of one token of ``tensor``; 0 when they cannot be read."""
    try:
        return math.prod(int(size) for size in tensor.shape[1:]) * int(tensor.element_size())
    except Exception:
        return 0


def forget() -> None:
    """Session state cleared, for tests."""
    global _host
    _host = UNKNOWN
    _said.clear()
