"""Krea Creative Mode: one explicit press, one creative roll, one image.

Creative Mode sits in front of Forge's Generate button and in front of LLM
Studio's *Generate Krea Prompt*, and it does the same thing in both places:

    source prompt
        -> resolve the Creative seed
        -> local Creative Director picks the art direction   (no model)
        -> exactly ONE Krea writer request
        -> one expanded Krea prompt
        -> pinned LoRA tags appended                          (txt2img only)
        -> native Forge generation                            (txt2img only)

Nothing here starts on its own. There is no idle timer, no observer on the
prompt box, no repeat loop and no state machine counting down to anything. A
roll happens because somebody pressed a button, and the button they pressed is
the one they already knew about.

What replaced what
------------------
This module is the successor to the Krea Live controller, and it is smaller by
most of a file. Live's cache, revision counter, cooperative cancellation of
in-flight writes, reroll scheduler and failure circuit breaker all existed to
manage work that started without being asked for. When every roll is explicit,
all of that is answerable by "the user pressed it again": there is nothing to
debounce, nothing to invalidate, and no loop that could run away.

One thing did survive, and it is the one that was never about Live: the arming
token. Forge's processing hook still needs a way to be handed an expansion that
was computed before the image job began, and it still must be impossible for a
nested or queued generation to spend that permission twice.

Why the model is asked before the image job starts
--------------------------------------------------
``mc_llm_sessions`` takes the broker's workload lock for a whole run and waits
while the host is generating, so an LLM completion and a diffusion pass never
overlap. Asking for an expansion from inside a Forge processing hook would
therefore be an LLM run waiting for the image job that is waiting for it. So
this module *requests* the expansion, from a Gradio handler, before anything
native has started, and the processing hook only ever *applies* one that already
exists. :meth:`Creative.consume` is the whole of the hook's vocabulary.

Three prompts, and only one of them is on screen
------------------------------------------------
``source`` is what the user typed and keeps editing; it stays in the txt2img box
untouched. ``expanded`` is the writer's one result. ``generation`` is that plus
the pinned LoRA tags, and it exists only between :meth:`Creative.consume` and
the line in the processing hook that assigns it.
"""

from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass, field

import mc_llm_sessions as sessions
import mc_llm_state

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

IDLE = "Idle"
DIRECTING = "Choosing the art direction"
WRITING = "Writing the Krea prompt"
READY = "Prompt ready"
ERROR = "Error"


# --------------------------------------------------------------------------- #
# The pinned LoRAs
# --------------------------------------------------------------------------- #


def pinned_tags(text) -> list[str]:
    """The extra-network tags in ``text``, and nothing else that was in it.

    The pinned-LoRA field is a text box, and a text box next to a prompt box is
    a text box somebody will eventually type prose into. Prose there would be
    prompt text reaching the image model without ever passing the writer -- a
    second, invisible prompt input. So the field is *parsed*, not appended: only
    ``<...>`` tags survive it.
    """
    import mc_lora

    return [match.group(0) for match in mc_lora.RE_EXTRA_NET.finditer(str(text or ""))]


def lora_suffix(text) -> str:
    """The pinned tags as one trailing fragment, or an empty string."""
    return " ".join(pinned_tags(text))


def generation_prompt(expanded: str, loras) -> str:
    """The expanded prompt with the pinned tags after it.

    The order is the order a hand-written Forge prompt uses: the sentence first,
    the networks at the end, so a prompt read back out of PNG metadata looks like
    something a person could have typed.
    """
    suffix = lora_suffix(loras)
    body = str(expanded or "").strip()
    return f"{body} {suffix}".strip() if suffix else body


# --------------------------------------------------------------------------- #
# One roll's result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Roll:
    """Everything one creative roll produced, and what it took to produce it."""

    recipe: object
    source: str
    expanded: str

    @property
    def creativity(self) -> int:
        return int(getattr(self.recipe, "creativity", 0))

    @property
    def creative_seed(self) -> int:
        return int(getattr(self.recipe, "creative_seed", 0))

    @property
    def llm_seed(self) -> int:
        return int(getattr(self.recipe, "llm_seed", 0))


@dataclass(frozen=True)
class Armed:
    """Permission for exactly one native generation to use one roll.

    The token is the permission; the rest is what the processing hook needs in
    order to act on it -- the prompt to substitute, and the values to write into
    infotext so the image can be explained afterwards.
    """

    token: str
    roll: Roll
    generation: str
    loras: str

    @property
    def metadata(self) -> dict:
        """What this generation records about how its prompt was written.

        The source phrase, the position, both seeds and the recipe as compact
        ids: everything needed to roll it again, and nothing that is already in
        the file. The expanded prompt is absent because it *is* the image's own
        ``Prompt:`` line -- writing the paragraph a second time under a key of
        ours would add a few hundred bytes to every PNG to repeat what the file
        already says, and create a copy a later paste could disagree with.
        """
        import mc_infotext

        recipe = self.roll.recipe
        recorded = {
            mc_infotext.CREATIVE_MODE: "True",
            mc_infotext.CREATIVE_CREATIVITY: self.roll.creativity,
            mc_infotext.CREATIVE_SEED: self.roll.creative_seed,
            mc_infotext.CREATIVE_LLM_SEED: self.roll.llm_seed,
            mc_infotext.CREATIVE_SOURCE: self.roll.source,
        }
        compact = getattr(recipe, "compact", "")
        if compact:
            recorded[mc_infotext.CREATIVE_RECIPE] = compact
        version = getattr(recipe, "library_version", "")
        if version:
            recorded[mc_infotext.CREATIVE_LIBRARY] = version
        if self.loras:
            recorded[mc_infotext.CREATIVE_LORAS] = self.loras
        return recorded


# --------------------------------------------------------------------------- #
# The checkpoint guard
# --------------------------------------------------------------------------- #


def checkpoint_objection() -> str:
    """Why Creative Mode must not arm against the selected checkpoint, or "".

    Creative Mode writes Krea 2 prompts: long, natural-language, written to
    Krea's own guidance. Handing one to SD 1.5 is not a smaller version of the
    feature, it is a paragraph fed to a model with 77 tokens of room, and the
    result would look like the extension was broken rather than like a choice.

    An architecture that cannot be identified is allowed through without
    complaint. Detection reads a checkpoint header and genuinely cannot see
    inside every GGUF or repacked build, so refusing on "unknown" would refuse
    real Krea 2 checkpoints -- and a guard that blocks the thing it exists to
    protect is worse than no guard.

    This is a txt2img concern only. LLM Studio writes prompts and generates no
    images, so which checkpoint happens to be loaded there is none of its
    business.
    """
    try:
        import mc_arch
        from modules import shared

        found = mc_arch.detect_loaded_engine()
        if found is mc_arch.UNKNOWN:
            found = mc_arch.detect_from_checkpoint_name(shared.opts.sd_model_checkpoint)
    except Exception:
        logger.debug("Model Chain: could not identify the image checkpoint for Creative Mode",
                     exc_info=True)
        return ""

    if found is mc_arch.UNKNOWN or found.key == "krea2":
        return ""
    return (f"Creative Mode writes Krea 2 prompts, and the selected checkpoint is "
            f"{found.label}. Select a Krea 2 checkpoint, or turn Creative Mode off to "
            "generate with the prompt as you typed it.")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

ENABLED = "krea_creative_enabled"
CREATIVITY = "krea_creativity"
SEED = "krea_creative_seed"
ANTI_REPETITION = "krea_creative_anti_repetition"
AXIS_MODES = "krea_creative_axis_modes"
FIXED_VALUES = "krea_creative_fixed"
LORAS = "krea_creative_loras"
HISTORY = "krea_creative_history"


def settings() -> dict:
    """Every Creative Mode preference, with the package's defaults filled in.

    One set of settings for both surfaces, not two. The axes, the Creativity
    position and the seed describe *how this installation does art direction*,
    and a user who has spent five minutes configuring ten axes in LLM Studio
    should not have to do it again in txt2img. The library's own ``defaults.json``
    supplies anything the file has never held.
    """
    from prompt_master.krea import director, variation

    try:
        stored = mc_llm_state.preferences()
    except Exception:
        logger.debug("Model Chain: could not read the Creative Mode preferences",
                     exc_info=True)
        stored = {}

    try:
        from prompt_master.krea import library as library_module

        defaults = dict(library_module.library().defaults)
        axis_keys = library_module.library().axis_keys
    except Exception:
        logger.debug("Model Chain: the creativity library could not be read", exc_info=True)
        defaults, axis_keys = {}, ()

    modes = dict(defaults.get("axis_modes") or {})
    modes.update({key: str(value).casefold()
                  for key, value in (stored.get(AXIS_MODES) or {}).items()
                  if str(value).casefold() in director.MODES})
    fixed = dict(defaults.get("fixed_values") or {})
    fixed.update({key: str(value) for key, value in (stored.get(FIXED_VALUES) or {}).items()
                  if value})

    return {
        "enabled": bool(stored.get(ENABLED, defaults.get("creative_mode_enabled", False))),
        "creativity": variation.clamp(stored.get(CREATIVITY,
                                                 defaults.get("creativity", variation.DEFAULT))),
        "seed": _seed(stored.get(SEED, defaults.get("creative_seed", director.RANDOM_SEED))),
        "anti_repetition": bool(stored.get(ANTI_REPETITION,
                                           defaults.get("anti_repetition", True))),
        "axis_modes": {key: modes.get(key, director.VARY) for key in axis_keys},
        "fixed_values": {key: value for key, value in fixed.items() if key in axis_keys},
        "loras": lora_suffix(stored.get(LORAS, "")),
    }


def _seed(value) -> int:
    from prompt_master.krea import director

    try:
        return int(value)
    except (TypeError, ValueError):
        return director.RANDOM_SEED


def remember(**values) -> None:
    """Keep a Creative Mode preference. Never fatal: this is a convenience."""
    try:
        mc_llm_state.remember(**values)
    except Exception:
        logger.debug("Model Chain: could not save the Creative Mode preferences",
                     exc_info=True)


def axis_settings(stored=None) -> dict:
    """The stored modes and pins as the Director's own type."""
    from prompt_master.krea import director

    stored = stored or settings()
    fixed = stored.get("fixed_values") or {}
    return {key: director.AxisSetting(mode=mode, fixed_id=fixed.get(key))
            for key, mode in (stored.get("axis_modes") or {}).items()}


# --------------------------------------------------------------------------- #
# What the last few rolls used
# --------------------------------------------------------------------------- #


def history() -> list[list[str]]:
    """The variant ids of the last few rolls, newest last.

    Ids and not prompts. The design is explicit about this and it is worth
    keeping explicit in the code: what anti-repetition needs to know is "did we
    just use the impasto medium", and storing the prompts to answer that would
    be keeping a transcript of everything anybody asked for in a preferences
    file.
    """
    try:
        stored = mc_llm_state.preferences().get(HISTORY) or []
    except Exception:
        return []
    rolls = []
    for entry in stored:
        if isinstance(entry, (list, tuple)):
            rolls.append([str(identifier) for identifier in entry])
    return rolls


def recent_ids() -> tuple[str, ...]:
    """Every id in the remembered history, flattened, for the Director."""
    return tuple(identifier for roll in history() for identifier in roll)


def note_roll(recipe) -> None:
    """Remember what one roll used, and forget the oldest one past the limit."""
    identifiers = [item.variant_id for item in getattr(recipe, "items", ())]
    if not identifiers:
        return
    try:
        from prompt_master.krea import library as library_module

        limit = library_module.library().anti_repetition.history_length
    except Exception:
        limit = 8
    remember(**{HISTORY: (history() + [identifiers])[-limit:]})


def forget_history() -> None:
    """Drop the recent-roll memory, so the next roll may pick anything again."""
    remember(**{HISTORY: []})


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class Creative:
    """Creative Mode's state for one WebUI. Two fields and a lock.

    Compare the module this replaced, which had a cache, a revision counter, a
    cancellation handle, a failure count and a status machine. All of those
    existed to manage work nobody had asked for. What is left is the last roll
    (so the diagnostic view has something to show) and the arming token (so one
    expansion makes exactly one image).
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._armed: Armed | None = None
        self._last: Roll | None = None
        self._status = IDLE

    @property
    def status(self) -> str:
        return self._status

    def say(self, status: str) -> str:
        self._status = status
        return status

    @property
    def last(self) -> Roll | None:
        with self._lock:
            return self._last

    @property
    def armed(self) -> Armed | None:
        """The pending arming, without consuming it. For status only."""
        with self._lock:
            return self._armed

    def disarm(self) -> None:
        """Throw away any unspent permission.

        Called when Creative Mode is switched off. Without it, turning the
        feature off between an expansion and a Generate click would leave a
        token that the next ordinary generation would spend.
        """
        with self._lock:
            self._armed = None
        self.say(IDLE)

    def arm(self, roll: Roll, loras) -> Armed:
        """Give one native generation permission to use ``roll``.

        Re-arming replaces any unused token rather than adding a second one.
        There is one Generate button and one browser; two live tokens would mean
        one of them belonged to a generation that never happened, waiting to be
        picked up by one that was about to be armed properly anyway.
        """
        armed = Armed(token=secrets.token_hex(8), roll=roll,
                      generation=generation_prompt(roll.expanded, loras),
                      loras=lora_suffix(loras))
        with self._lock:
            self._armed = armed
            self._last = roll
        return armed

    def consume(self, token=None) -> Armed | None:
        """Take the arming token, once. The processing hook's whole vocabulary.

        Returns ``None`` for a generation this module did not arm -- an ordinary
        Generate with Creative Mode off, Stage 2's own nested
        ``process_images`` call, a queued request from a browser tab that has
        since been closed -- and ``None`` is the answer that leaves Forge's
        prompt exactly as the user typed it.
        """
        with self._lock:
            armed, self._armed = self._armed, None
        if armed is None:
            return None
        if token and token != armed.token:
            return None
        return armed

    # -- one roll ---------------------------------------------------------- #

    def roll(self, source: str, stored=None, references=(), guard_checkpoint=False):
        """One creative roll: direct locally, then ask the model once.

        Yields :class:`mc_llm_sessions.Event` throughout so a caller can put the
        writer's progress on screen, and finishes with ``DONE`` carrying the
        expanded prompt. The recipe is on :attr:`last` by then.

        The Director runs first and runs entirely in this process. By the time
        the writer is called, every creative decision has already been made and
        written down; the model's whole job is to turn one brief into one Krea
        prompt. That ordering is what makes "exactly one model call" a property
        of the design rather than a thing to be careful about.
        """
        from prompt_master.core.models import draw_seed
        from prompt_master.krea import director

        source = str(source or "").strip()
        if not source:
            yield sessions.Event(sessions.FAILED,
                                 "Type what you want in the prompt box first.")
            return

        if guard_checkpoint:
            objection = checkpoint_objection()
            if objection:
                yield sessions.Event(sessions.FAILED, objection)
                return

        stored = stored or settings()
        self.say(DIRECTING)
        try:
            recipe = director.roll(
                source=source,
                creativity=stored["creativity"],
                creative_seed=stored["seed"],
                settings=axis_settings(stored),
                history=recent_ids() if stored.get("anti_repetition") else ())
        except Exception as exc:
            # A library that will not load is a Creative Mode that cannot
            # direct. It is emphatically not a reason to send the request
            # anyway with no brief: the user asked for art direction, and a
            # plain expansion pretending to be one is the wrong answer.
            self.say(ERROR)
            logger.debug("Model Chain: the Creative Director failed", exc_info=True)
            yield sessions.Event(sessions.FAILED, f"The creativity library could not be "
                                                  f"used: {exc}")
            return

        yield sessions.Event(sessions.STATUS, _directed(recipe))
        self.say(WRITING)

        cancel = sessions.Cancellation()
        # Held by name and closed explicitly: the loop below stops as soon as it
        # has the finished prompt, and the statement that gives the GPU back is
        # in that run's ``finally``. Closing it is what runs that finally now
        # rather than whenever the interpreter next collects the frame.
        run = sessions.krea(source, list(references or []), recipe.llm_seed, cancel,
                            recipe.creativity, recipe.brief)
        written = ""
        try:
            for event in run:
                if event.kind == sessions.DONE:
                    written = event.text
                    break
                if event.kind in (sessions.FAILED, sessions.CANCELLED):
                    self.say(ERROR if event.kind == sessions.FAILED else IDLE)
                    yield event
                    return
                yield event
        except Exception as exc:
            self.say(ERROR)
            logger.debug("Model Chain: the Creative Mode roll failed", exc_info=True)
            yield sessions.Event(sessions.FAILED, str(exc))
            return
        finally:
            run.close()

        if not written.strip():
            self.say(ERROR)
            yield sessions.Event(sessions.FAILED, "The model returned an empty prompt.")
            return

        with self._lock:
            self._last = Roll(recipe=recipe, source=source, expanded=written.strip())
        if stored.get("anti_repetition"):
            note_roll(recipe)
        self.say(READY)
        yield sessions.Event(sessions.DONE, written.strip())


def _directed(recipe) -> str:
    """One line saying what the Director chose, for the status area.

    The count and the seed, not the brief. The brief is shown in full in the
    diagnostic view where somebody has asked to see it; a status line that
    printed nine sentences of art direction would push everything else off the
    panel every time somebody pressed Generate.
    """
    count = len(getattr(recipe, "items", ()))
    if not count:
        return f"No creative direction at Creativity {recipe.creativity}."
    return (f"Directed {count} {'axis' if count == 1 else 'axes'} · "
            f"Creative seed {recipe.creative_seed}")


creative = Creative()
"""The one Creative Mode session.

A module singleton for the reason ``mc_llm_runtime.runtime`` and the broker are:
one browser, one GPU, one llama-server, one Forge. The arming token is what
keeps that honest across the gap between the Gradio handler that writes it and
the Forge thread that reads it.
"""
