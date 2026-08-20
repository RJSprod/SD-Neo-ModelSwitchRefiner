"""Krea Live: one expansion per prompt, reused for every image drawn from it.

Krea Live is not a tab and not a second image backend. It is one gate placed in
front of Forge's own Generate button:

    Live off:  positive prompt ------------------------> Forge
    Live on:   positive prompt -> Krea writer -> prompt -> Forge

Everything after the arrow on the right is unchanged Forge -- the checkpoint,
the sampler, the size, the seed, the LoRAs, the extensions, the saving, the
gallery. What this module owns is the short stretch on the left, and the one
rule that stretch has to keep.

The rule
--------
**One LLM call per new prompt-authoring state.** Not one per Generate, and
emphatically not one per image. A prompt-authoring state is the set of inputs
that actually reach the language model:

    the source prompt, Creativity, the prompt seed policy, and which model is
    loaded to write with

and nothing else. Steps, image seed, width, height, sampler, CFG and pinned
LoRAs are all excluded, because none of them is sent to the writer and a reroll
that pinged the LLM again would be a reroll that cost a GPU handover and eight
seconds to draw the same prompt twice. :func:`cache_key` is that sentence in
code, and the excluded list is what makes rerolling free.

So the cache is not an optimisation that could be dropped. It is the mechanism
the whole feature is built on, and the test that a reroll makes no LLM call is
a test of the product rather than of a cache.

Why the LLM has to run before the image job starts
--------------------------------------------------
``mc_llm_sessions`` takes the broker's workload lock for a whole run and waits
while the host is generating, so that an LLM completion and a diffusion pass
never overlap. Asking for an expansion from inside a Forge processing hook
would therefore mean an image job waiting on an LLM run that is itself waiting
for the image job to finish. That is why this module *requests* an expansion --
from a Gradio handler, before anything native has started -- and the processing
hook only ever *applies* one that already exists. :func:`Live.consume` is the
whole of the hook's vocabulary.

Three prompts, and only one of them is on screen
------------------------------------------------
``source`` is what the user typed and keeps editing; it stays in the txt2img
box untouched, because a box that fills with a 400-word paragraph the moment
you stop typing is a box you cannot iterate in. ``expanded`` is the writer's
one result. ``generation`` is that plus the pinned LoRA tags, and it exists
only between :func:`Live.consume` and the line in the processing hook that
assigns it. The user never types in it and never has to look at it.

State, and why a module-level one is right here
-----------------------------------------------
One browser, one GPU, one llama-server, one Forge -- the same assumption
``mc_llm_runtime.runtime`` and the broker already make, and the reason a Live
session is a module singleton rather than a Gradio ``State``. The arming token
is what keeps that honest across the gap between the handler that writes it and
the hook that reads it: it is drawn fresh per arming and consumed on first use,
so a queued second generation, a nested ``process_images`` call or a stale
browser cannot make a second image out of one expansion's permission.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field

import mc_llm_sessions as sessions
import mc_llm_state

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

# --------------------------------------------------------------------------- #
# What the strip says it is doing
# --------------------------------------------------------------------------- #

OFF = "Off"
WAITING = "Waiting"
WRITING = "Writing prompt"
READY = "Prompt ready"
GENERATING = "Generating"
PENDING = "New text pending"
REROLLING = "Rerolling"
STOPPING = "Stopping"
ERROR = "Error"

DEFAULT_DELAY = 5.0
MIN_DELAY = 0.5
MAX_DELAY = 30.0
"""The idle debounce, in seconds.

Five because it is long enough to type a sentence through without tripping it
and short enough that stopping to think produces a prompt rather than a wait.
The floor is not zero: a debounce of zero is not a faster Live mode, it is one
LLM call per keystroke.
"""

FAILURE_LIMIT = 3
"""Consecutive failures before automatic repetition gives up.

A reroll loop whose generation fails and which rerolls again is a loop that
fails at whatever rate the failure happens at, forever, and the first anybody
knows of it is the log file. Three is enough to ride out one bad moment and few
enough that a broken configuration stops being retried while somebody is still
looking at the screen.
"""


# --------------------------------------------------------------------------- #
# The pinned LoRAs
# --------------------------------------------------------------------------- #


def pinned_tags(text) -> list[str]:
    """The extra-network tags in ``text``, and nothing else that was in it.

    The pinned-LoRA field is a text box, and a text box next to a prompt box is
    a text box somebody will eventually type prose into. Prose there would be
    prompt text that reached the image model without ever passing the writer --
    a second, invisible prompt input, which is exactly what the design rules
    out by saying the LLM receives none of these tags and the image model
    receives nothing else from this field. So the field is *parsed*, not
    appended: only ``<...>`` tags survive it.
    """
    import mc_lora

    return [match.group(0) for match in mc_lora.RE_EXTRA_NET.finditer(str(text or ""))]


def lora_suffix(text) -> str:
    """The pinned tags as one trailing fragment, or an empty string."""
    return " ".join(pinned_tags(text))


def generation_prompt(expanded: str, loras) -> str:
    """The expanded prompt with the pinned tags after it.

    The order is deliberate and is the order a hand-written Forge prompt uses:
    the sentence first, the networks at the end, so that a prompt read back out
    of PNG metadata looks like something a person could have typed.
    """
    suffix = lora_suffix(loras)
    body = str(expanded or "").strip()
    if not suffix:
        return body
    return f"{body} {suffix}".strip()


# --------------------------------------------------------------------------- #
# The prompt-authoring state
# --------------------------------------------------------------------------- #


def writer_identity() -> str:
    """Which model would write the prompt, as a string that changes when it does.

    Part of the cache key because the expansion is that model's work. Swapping
    the LLM in Setup and pressing Generate must produce a new prompt rather
    than replaying the previous model's, and no user is going to retype their
    source text to make that happen.
    """
    try:
        import mc_llm_runtime

        return str(mc_llm_runtime.config().model or "")
    except Exception:
        logger.debug("Model Chain: could not identify the LLM for the Live cache key",
                     exc_info=True)
        return ""


def cache_key(source: str, creativity, prompt_seed, identity=None) -> str:
    """A digest of everything the writer is actually given.

    Deliberately short and deliberately total: every input the LLM sees is in
    here, and everything else in the extension is not. That is the difference
    between "changing Steps re-runs the LLM" and "changing Steps draws another
    image", and it is settled here rather than at each of the call sites that
    might otherwise each decide for themselves.

    A fixed prompt seed is part of the key; the random sentinel is too, as
    itself. That is correct rather than a compromise: with the source and
    Creativity unchanged, "random" has already been resolved into one concrete
    seed stored beside the cached expansion, and re-drawing it would be a
    second LLM call for a prompt nobody asked to have rewritten. Editing the
    source, or moving Creativity, is how a user asks for that.
    """
    from prompt_master.krea.variation import clamp

    material = "\x1f".join((
        str(source or "").strip(),
        str(clamp(creativity)),
        str(int(prompt_seed)),
        writer_identity() if identity is None else str(identity),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class Expansion:
    """One accepted Krea prompt, and the state it was written from.

    Frozen, because a cached expansion that could be edited in place is a cache
    whose key has stopped describing its contents -- and the key is the only
    thing standing between a reroll and another LLM call.
    """

    key: str
    source: str
    expanded: str
    creativity: int
    prompt_seed: int
    created: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Armed:
    """Permission for exactly one native generation to use one expansion.

    The token is the permission; the rest of the record is what the processing
    hook needs in order to act on it -- the prompt to substitute, and the values
    to write into infotext so the generation can be explained afterwards.
    """

    token: str
    expansion: Expansion
    generation: str
    loras: str

    @property
    def metadata(self) -> dict:
        """What this generation records about how its prompt was written.

        The source, the position and the prompt seed: the three things that are
        *not* recoverable from the finished infotext, since the expanded prompt
        is already in it as the generation's own prompt line. Writing that
        paragraph a second time under a Krea Live key would add a few hundred
        bytes to every PNG in order to repeat what the file already says.

        The pinned tags are recorded because they are the difference between
        the writer's result and the prompt that was generated from: with them
        written down the expansion can be recovered from the infotext exactly,
        which is what makes the pair reproducible.
        """
        import mc_infotext

        recorded = {mc_infotext.LIVE_SOURCE: self.expansion.source,
                    mc_infotext.LIVE_CREATIVITY: self.expansion.creativity,
                    mc_infotext.LIVE_PROMPT_SEED: self.expansion.prompt_seed}
        if self.loras:
            recorded[mc_infotext.LIVE_LORAS] = self.loras
        return recorded


# --------------------------------------------------------------------------- #
# The checkpoint guard
# --------------------------------------------------------------------------- #


def checkpoint_objection() -> str:
    """Why Live must not arm against the selected checkpoint, or an empty string.

    Krea Live writes Krea 2 prompts: long, natural-language, written to Krea's
    own guidance. Handing one to SD 1.5 is not a smaller version of the feature,
    it is a paragraph fed to a model with 77 tokens of room, and the result
    would look like the extension was broken rather than like a choice.

    An architecture that cannot be identified is allowed through without
    complaint. Detection reads a checkpoint header and genuinely cannot see
    inside every GGUF or repacked build, so refusing on "unknown" would refuse
    real Krea 2 checkpoints -- and a guard that blocks the thing it exists to
    protect is worse than no guard.
    """
    try:
        import mc_arch
        from modules import shared

        found = mc_arch.detect_loaded_engine()
        if found is mc_arch.UNKNOWN:
            found = mc_arch.detect_from_checkpoint_name(shared.opts.sd_model_checkpoint)
    except Exception:
        logger.debug("Model Chain: could not identify the image checkpoint for Krea Live",
                     exc_info=True)
        return ""

    if found is mc_arch.UNKNOWN or found.key == "krea2":
        return ""
    return (f"Krea Live writes Krea 2 prompts, and the selected checkpoint is "
            f"{found.label}. Select a Krea 2 checkpoint, or turn Krea Live off to "
            "generate with the prompt as you typed it.")


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class Live:
    """Krea Live's whole state, for one WebUI.

    Two things are protected by the lock and nothing else needs to be: the
    cached expansion and the arming token. Both are written by a Gradio handler
    thread and read by whichever thread Forge runs a generation on, and both
    have a failure mode -- two expansions racing to arm, or one token consumed
    twice -- that is invisible until it produces an image with the wrong prompt.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Expansion | None = None
        self._armed: Armed | None = None
        self._cancel: sessions.Cancellation | None = None
        # The newest source revision anybody has told us about. A run started
        # at revision 4 and finishing after 5 has arrived is a run whose answer
        # is about text nobody is looking at any more.
        self._revision = 0
        self._failures = 0
        self._status = OFF

    # -- what the strip shows -------------------------------------------- #

    @property
    def status(self) -> str:
        return self._status

    def say(self, status: str) -> str:
        self._status = status
        return status

    @property
    def cached(self) -> Expansion | None:
        with self._lock:
            return self._cache

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def exhausted(self) -> bool:
        """Whether automatic repetition has given up."""
        return self._failures >= FAILURE_LIMIT

    # -- staleness -------------------------------------------------------- #

    def revise(self) -> int:
        """Record that the source text has moved on, and cancel a run about the old one.

        Called when the browser's debounce sees an edit arrive while an
        expansion is in flight -- once per typing burst, not once per keystroke,
        because the browser does the counting and only speaks when it has
        something to cancel.

        Cancelling is cooperative and the late answer is discarded regardless:
        :meth:`prepare` re-checks the revision it started at before it caches or
        arms anything, so a run that cannot be stopped in time still cannot
        reach Forge.
        """
        with self._lock:
            self._revision += 1
            self._cache = None
            self._armed = None
            if self._cancel is not None:
                self._cancel.cancel()
            return self._revision

    def stop(self) -> None:
        """Stop Live: no more debounce, no more rerolls, cancel the writer.

        What is deliberately *not* here is any attempt to stop a diffusion pass
        that has already started. Forge owns that, it has its own Interrupt, and
        a Live stop that reached into a running sampler would be this extension
        deciding to spoil an image somebody may well want to keep.
        """
        with self._lock:
            self._revision += 1
            self._armed = None
            if self._cancel is not None:
                self._cancel.cancel()
        self.say(OFF)

    def reset_failures(self) -> None:
        self._failures = 0

    def note_failure(self) -> int:
        """Count a failed automatic cycle, so repetition can give up."""
        self._failures += 1
        return self._failures

    # -- arming ----------------------------------------------------------- #

    def arm(self, expansion: Expansion, loras) -> Armed:
        """Give one native generation permission to use ``expansion``.

        Re-arming replaces any unused token rather than adding a second one.
        There is one Generate button and one browser; two live tokens would mean
        one of them belonged to a generation that never happened, waiting to be
        picked up by one that was about to be armed properly anyway.
        """
        armed = Armed(token=secrets.token_hex(8), expansion=expansion,
                      generation=generation_prompt(expansion.expanded, loras),
                      loras=lora_suffix(loras))
        with self._lock:
            self._armed = armed
        return armed

    def consume(self, token=None) -> Armed | None:
        """Take the arming token, once. The processing hook's whole vocabulary.

        Returns ``None`` for a generation this module did not arm -- an ordinary
        Generate with Live off, Stage 2's own nested ``process_images`` call, a
        queued request from a browser tab that has since been closed -- and
        ``None`` is the answer that leaves Forge's prompt exactly as the user
        typed it.

        ``token`` is checked when one is supplied, so that a caller holding a
        specific arming can tell it apart from a newer one. The processing hook
        does not supply it: by the time it runs, the newest arming is by
        definition the one the Generate click carried.
        """
        with self._lock:
            armed, self._armed = self._armed, None
        if armed is None:
            return None
        if token and token != armed.token:
            return None
        return armed

    @property
    def armed(self) -> Armed | None:
        """The pending arming, without consuming it. For status only."""
        with self._lock:
            return self._armed

    # -- the one call ------------------------------------------------------ #

    def prepare(self, source, creativity, loras, prompt_seed):
        """Make sure an expansion for this state exists, and arm it.

        Yields :class:`mc_llm_sessions.Event` throughout, so the caller can put
        the writer's progress on the strip as it happens, and finishes with
        ``DONE`` carrying the arming token.

        The whole one-call rule lives in the first branch: when the key matches
        what is cached, no client is contacted, no GPU is asked for and nothing
        is said about writing anything -- the cached prompt is armed and the
        generator finishes. Every reroll, every Steps change, every LoRA weight
        tweak and every new image seed takes that branch.
        """
        from prompt_master.core.models import RANDOM_SEED, draw_seed
        from prompt_master.krea.variation import clamp

        source = str(source or "").strip()
        position = clamp(creativity)
        requested_seed = int(prompt_seed if prompt_seed is not None else RANDOM_SEED)

        if not source:
            yield sessions.Event(sessions.FAILED,
                                 "Type what you want in the positive prompt first.")
            return

        objection = checkpoint_objection()
        if objection:
            yield sessions.Event(sessions.FAILED, objection)
            return

        key = cache_key(source, position, requested_seed)
        with self._lock:
            cached = self._cache if self._cache and self._cache.key == key else None
        if cached is not None:
            armed = self.arm(cached, loras)
            self.reset_failures()
            self.say(READY)
            yield sessions.Event(sessions.DONE, armed.token)
            return

        started_at = self._revision
        cancel = sessions.Cancellation()
        with self._lock:
            self._cancel = cancel
        resolved_seed = draw_seed() if requested_seed == RANDOM_SEED else requested_seed
        self.say(WRITING)

        # Held by name and closed explicitly. Every path out of the loop below
        # except exhaustion leaves the run suspended mid-``yield``, and the
        # statement that gives the GPU back is in that run's ``finally`` --
        # closing it is what runs that finally now rather than whenever the
        # interpreter next collects the frame. A workload lock released "soon"
        # is a lock the next image generation waits on for no reason.
        run = sessions.krea(source, [], resolved_seed, cancel, position)
        written = ""
        try:
            for event in run:
                if event.kind == sessions.DONE:
                    written = event.text
                    break
                if event.kind in (sessions.FAILED, sessions.CANCELLED):
                    self.note_failure()
                    self.say(ERROR if event.kind == sessions.FAILED else OFF)
                    yield event
                    return
                yield event
        except Exception as exc:
            self.note_failure()
            self.say(ERROR)
            logger.debug("Model Chain: the Krea Live expansion failed", exc_info=True)
            yield sessions.Event(sessions.FAILED, str(exc))
            return
        finally:
            run.close()
            with self._lock:
                if self._cancel is cancel:
                    self._cancel = None

        # The answer arrived; the question may have changed while it was being
        # written. A late expansion is discarded here and never reaches Forge --
        # the browser has already debounced the newer text and will ask for it
        # in a moment, and one call for that state is what it will get.
        if self._revision != started_at:
            self.say(PENDING)
            yield sessions.Event(sessions.CANCELLED,
                                 "The prompt changed while that was being written.")
            return

        if not written.strip():
            self.note_failure()
            self.say(ERROR)
            yield sessions.Event(sessions.FAILED, "The model returned an empty prompt.")
            return

        expansion = Expansion(key=key, source=source, expanded=written.strip(),
                              creativity=position, prompt_seed=resolved_seed)
        with self._lock:
            self._cache = expansion
        armed = self.arm(expansion, loras)
        self.reset_failures()
        self.say(READY)
        yield sessions.Event(sessions.DONE, armed.token)


live = Live()
"""The one Live session. See the module docstring for why it is a singleton."""


# --------------------------------------------------------------------------- #
# Remembered positions
# --------------------------------------------------------------------------- #


def remembered() -> dict:
    """Where the Live strip was left: creativity, delay, pinned LoRAs, prompt seed.

    Creativity is stored under its own key, separate from LLM Studio's. The
    positions differ because the tasks differ -- authoring a prompt to keep is
    not iterating images every five seconds -- while the *meaning* of a value is
    one shared table in ``prompt_master.krea.variation`` and is not duplicated
    here.
    """
    from prompt_master.krea.variation import DEFAULT, clamp

    try:
        stored = mc_llm_state.preferences()
    except Exception:
        logger.debug("Model Chain: could not read the Krea Live preferences", exc_info=True)
        stored = {}
    try:
        delay = float(stored.get("krea_live_delay", DEFAULT_DELAY))
    except (TypeError, ValueError):
        delay = DEFAULT_DELAY
    try:
        seed = int(stored.get("krea_live_seed", -1))
    except (TypeError, ValueError):
        seed = -1
    return {"creativity": clamp(stored.get("krea_live_creativity", DEFAULT)),
            "delay": max(MIN_DELAY, min(MAX_DELAY, delay)),
            "loras": lora_suffix(stored.get("krea_live_loras", "")),
            "seed": seed}


def remember(**values) -> None:
    """Keep the Live strip's positions. Never fatal: this is a convenience."""
    try:
        mc_llm_state.remember(**{f"krea_live_{name}": value for name, value in values.items()
                                 if value is not None})
    except Exception:
        logger.debug("Model Chain: could not save the Krea Live preferences", exc_info=True)
