"""Neutralize Prompt: the stage that takes pose and placement out, and nothing else.

The first stage of the Image Pipeline, and the narrowest. Before Creative Mode
reads the prompt, this optionally removes two classes of constraint from it --
how a body is arranged, and where in the frame something sits -- and hands
what is left to every stage that follows. It removes geometry, not
characteristics; configuration, not activity; image placement, not world
location. It never adds a word, and what it hands on is proven to be a subset
of what it was given, by :mod:`prompt_master.krea.neutralizer` rather than by
trust.

What it is
----------
A consumer of the existing language-model platform, not a platform of its own.
The Neutralizer is the third first-class LLM role, and it inherits everything
the other two already solved: the installation's configuration until somebody
splits it, the same physical-card identity, the same runtime registry that
coalesces identically configured roles onto one llama-server, the same
workload lock, the same cancellation, the same progress bar. This module owns
what the Composer's does in :mod:`mc_spatial`: driving one pass, reading its
events, saying what happened, and describing the result to the image's
metadata.

What it may never do
--------------------
Move the image checkpoint. The reserve handed to the pass is a promise about
VRAM it will leave alone; it is not authority to reclaim any. When the
language model cannot be placed without disturbing the image side, the
placement negotiates on its own side -- context, residency, system RAM -- and
if that is not enough the pass fails and the prompt as typed answers. There is
no image-reclaim call in this module and none reachable from it, and a test
asserts as much under an artificial shortfall.

Two ways not to run, and they are not the same
----------------------------------------------
A server that will not start, a reply the guard refuses, a timeout, an empty
answer: these are a stage failure, and a stage failure is nonfatal. The source
is handed on exactly as typed, the console and the result say why, and
Creative and Spatial carry on as though the toggle had been off.

The user pressing Stop is different, and :class:`Neutralized` keeps the two
apart on purpose. An Interrupt during this pass is the generation being
cancelled, and the pipeline must not read it as "the neutralizer was
unavailable, generate from the source anyway". It cancels the pass
cooperatively, leaves residency as it found it, and lets the host's own
cancellation do what Stop has always done.
"""

from __future__ import annotations

import logging

import mc_llm_progress
import mc_llm_sessions as sessions
import mc_progress

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

STAGE = "neutralize"
"""The Image Pipeline row this stage owns, and the key the plan files it under."""

TITLE = "Neutralize Prompt"
"""What the stage is called on the pipeline. The *role* is the Pose Neutralizer."""


class Neutralized:
    """What the pass produced, or why it did not, in the form the pipeline reads.

    Three outcomes and three fields, because they lead three different places:
    ``text`` is a validated subtraction of the source and replaces it;
    ``failed`` is a reason the source stays as typed; ``stopped`` is the user's
    Interrupt, which replaces nothing and carries on to nothing.
    """

    def __init__(self, text: str = "", failed: str = "", stopped: bool = False,
                 removed: int = 0):
        self.text = str(text or "")
        self.failed = str(failed or "")
        self.stopped = bool(stopped)
        self.removed = int(removed)

    @property
    def ran(self) -> bool:
        """Whether the stage completed and its answer was accepted.

        True for an accepted reply that happens to equal the source: "ran"
        means the pass finished and the pipeline used what it said, not that
        the text changed. An image made that way records the stage as having
        run, because it did.
        """
        return not self.failed and not self.stopped


def neutralize(source: str, reserve: int = 0, task_id: str = "") -> Neutralized:
    """Run the Neutralizer once, and never raise.

    Returns the neutralized source, or the reason the source stands. Every
    failure -- a server that will not start, a reply the guard refused, a
    timeout, nothing back -- is a :class:`Neutralized` with ``failed`` set,
    and the pipeline's answer to every one of them is the prompt as typed.
    An optional copy-editor being unavailable is not a reason to refuse
    somebody the picture they asked for.

    ``stopped`` is the one outcome that is not a failure. The Interrupt button
    on the host's bar is read once per event, exactly as the Composer reads it,
    and when it has been pressed the pass is cancelled cooperatively and the
    answer says *stopped* rather than *failed* -- so the pipeline can tell a
    generation that is being cancelled from a stage that merely did not run.
    The flag itself is left as the user set it: the bar is the generation's,
    and the host reads the same flag a moment later to stop the sampling.

    ``reserve`` is VRAM the pass promises to leave alone for the image
    generation that follows; the caller sizes it from the same plan that
    protects the image phase. It constrains where the language model is placed
    and authorises nothing.
    """
    from prompt_master.krea import neutralizer

    source = str(source or "").strip()
    if not source:
        return Neutralized(failed="there was no transformable text to neutralize")

    cancel = sessions.Cancellation()
    run = sessions.krea_neutralize(source, neutralizer.SEED, cancel, reserve)
    progress = mc_llm_progress.reporter
    progress.begin(task_id, _size(source), warm=_warm(), claim=False,
                   kind=mc_llm_progress.NEUTRALIZER)
    result = Neutralized(failed="the neutralizer did not answer")
    written = ""
    finished = False
    try:
        for event in run:
            if event.kind == sessions.DONE:
                text = neutralizer.clean(event.text)
                # The session has already checked this; checked again here
                # because this is the function whose answer replaces a
                # prompt, and a subtraction is the one property that answer
                # must have whoever produced it.
                refused = neutralizer.subtraction_error(source, text)
                if refused:
                    result = Neutralized(failed=refused)
                else:
                    result = Neutralized(text=text, removed=neutralizer.removed(source, text))
                    written = text
                    finished = True
                break
            if event.kind == sessions.CANCELLED:
                result = Neutralized(stopped=True)
                break
            if event.kind == sessions.FAILED:
                result = Neutralized(failed=event.text or "the neutralizer failed")
                break
            if event.kind == sessions.CHUNK:
                progress.enter(mc_progress.PHASE_KREA_WRITE)
                progress.wrote(event.text)
            elif event.kind == sessions.STATUS:
                progress.enter(_phase_for(event.text))
            if progress.interrupted():
                cancel.cancel()
                result = Neutralized(stopped=True)
                break
    except Exception as exc:
        logger.debug("Model Chain: the prompt neutralizer failed", exc_info=True)
        result = Neutralized(failed=str(exc) or exc.__class__.__name__)
    finally:
        run.close()
        if finished:
            progress.end(written)
        else:
            progress.abandon()
    return result


def _size(source: str) -> int:
    """How much text the pass is about to evaluate, in characters.

    The whole request, instruction included. The Neutralizer's system message
    is not the one llama.cpp has cached for the writer -- it cannot be, the
    two say opposite things -- so on a server shared with Creative Mode it
    really is read on every run, and the bar prices it in. Once this pass has
    its own server, or runs first on a warm one, the measured rate corrects the
    guess.
    """
    from prompt_master.krea import neutralizer

    try:
        return len(neutralizer.system_prompt()) + len(neutralizer.user_content(source))
    except Exception:
        logger.debug("Model Chain: could not size the neutralizer's request", exc_info=True)
        return max(len(source), 1)


def _warm() -> bool:
    """Whether this role's server is already up, for the waiting phase's guess.

    The Neutralizer's own runtime, resolved the way the request will resolve
    it, because with a role split the module singleton may be somebody else's
    server. A wrong answer costs one run a poor estimate on the phase that is
    over before anybody reads it, so this guesses cold -- the pessimistic
    direction -- when it cannot tell.
    """
    try:
        import mc_llm_roles
        import mc_llm_runtime

        return bool(mc_llm_runtime.registry.for_role(mc_llm_roles.NEUTRALIZER).running())
    except Exception:
        return False


def _phase_for(status: str) -> str:
    """Which bar phase a status line means, for this pass.

    The session says what it is doing in prose; this maps that back onto a
    phase. Preparing and waiting are the wait; the pass's own announcement,
    emitted immediately before the request goes out, is the start of prompt
    evaluation.
    """
    text = str(status or "").casefold()
    if sessions.NEUTRALIZING.casefold() in text:
        return mc_progress.PHASE_KREA_READ
    return mc_progress.PHASE_KREA_WAIT


# --------------------------------------------------------------------------- #
# What a neutralized generation records about itself
# --------------------------------------------------------------------------- #


def metadata(source: str) -> dict:
    """The Neutralize keys for one generation's infotext.

    Written only when the pass ran and its answer was used -- a switched-on
    toggle whose pass failed records nothing, because the picture was made from
    the prompt as typed and a record saying otherwise would tell a later paste
    to switch off a stage that never touched it.

    The image's own ``Prompt:`` line stays the whole answer to *how do I make
    this again*; it is the finished prompt, neutralized and then whatever else
    ran. These two keys answer *how do I get back to what I typed*: the flag,
    so an ordinary paste can bypass the stage and an explicit restore can
    re-arm it, and the source exactly as typed, brackets and all, because a
    restore is supposed to hand back the prompt somebody wrote and a source
    with its constraints quietly missing would be a workflow that cannot be
    continued from.
    """
    import mc_infotext

    return {mc_infotext.NEUTRALIZE_MODE: "True",
            mc_infotext.NEUTRALIZE_SOURCE: str(source or "")}
