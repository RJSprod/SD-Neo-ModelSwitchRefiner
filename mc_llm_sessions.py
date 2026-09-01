"""The LLM modes, as streaming generators with no UI in them.

Section 4 requires Prompt Studio, Conversation and MiniMax to stay distinct
products rather than collapsing into one chat workflow, and section 17.B
requires the LTX business logic to come across without its Qt presentation
layer. Both are served by the same decision: the orchestration each mode's Qt
worker used to do lives here, in a form that has never heard of a widget. Krea
2 arrived later and under the same rule: it is a fourth generator here rather
than a persona, a variant or an option on somebody else's panel.

Each mode is a generator of :class:`Event`. The Qt version pushed the same
information out as signals -- ``status``, ``chunk``, ``positive_ready``,
``negative_ready``, ``captioned``, ``failed`` -- and the sequence of events
below is deliberately the same sequence, because that sequence *is* the
product. Prompt Studio still emits a positive and a negative separately;
MiniMax still emits its caption before its prompt; Krea emits one caption per
reference, in the order the user put them in; Conversation still emits one
stream of reply text. Nothing is merged into a single response blob, which is
the thing section 4.2 explicitly forbids.

Why generators
--------------
Gradio streams by iterating a generator, and llama.cpp streams by calling a
callback. :func:`_streamed` is the join between the two: it runs the callback
API on a worker thread and hands chunks back through a queue. The alternative
-- rewriting the vendored client to yield -- would have meant editing the one
file whose job is to be comparable against its source.

Serialization and cancellation
------------------------------
Every run takes the broker's workload lock for its whole duration (section 15),
so an LLM completion and an image pass take turns even when both models are
resident. While the lock is held by a generation, the run reports what it is
waiting for rather than appearing hung, and a cancel while waiting is honoured
without ever having started a server. Cancellation is cooperative and always
leaves residency exactly as it found it: the process stays up, the broker's
register is untouched, and the next request is a warm one.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import mc_broker
import mc_llm_roles
import mc_llm_runtime

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

# Event kinds.
STATUS = "status"
CHUNK = "chunk"
POSITIVE = "positive"
NEGATIVE = "negative"
CAPTION = "caption"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

WAIT_NOTICE_SECONDS = 0.75
"""How long to wait for the GPU before saying so.

Short enough that a user never wonders whether the button worked, long enough
that the common case -- the GPU is free, the lock is taken instantly -- does
not flash a message about waiting for something that was never busy.
"""

WAIT_POLL_SECONDS = 0.25


@dataclass
class Event:
    """One thing that happened during a run."""

    kind: str
    text: str = ""
    data: object = None
    """A structured result, for the one run whose answer is not a paragraph.

    The Spatial Composer finishes with two strings rather than one, and the
    alternative to a second field is encoding a pair into ``text`` at one end
    and decoding it at the other -- a private format between two functions in
    the same repository, which is a thing to get wrong rather than a thing to
    read. Every other mode leaves it ``None`` and every existing consumer reads
    ``text`` exactly as it did.
    """

    @property
    def terminal(self) -> bool:
        return self.kind in (DONE, FAILED, CANCELLED)


class Cancellation:
    """A run's stop button, in a form both this module and llama.cpp accept.

    Wraps ``threading.Event`` rather than being one so a UI can hold a handle
    that survives being handed between Gradio callbacks, and so the name of the
    thing reads as what it is at the call sites below.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def event(self) -> threading.Event:
        return self._event

    def is_set(self) -> bool:
        return self._event.is_set()


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #


_SENTINEL = object()


def _streamed(work):
    """Run ``work(on_text)`` on a thread and yield the text it produces.

    ``work`` is handed the callback llama.cpp's client expects and must return
    the complete text. The generator yields ``(chunk, None)`` for each piece
    and finally ``(None, result)`` -- or raises whatever ``work`` raised, on
    the consumer's thread, so an error surfaces where it can be reported
    instead of dying silently in a worker.
    """
    pipe: queue.Queue = queue.Queue()
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = work(pipe.put)
        except BaseException as exc:  # re-raised on the consumer's thread
            outcome["error"] = exc
        finally:
            pipe.put(_SENTINEL)

    worker = threading.Thread(target=run, name="model-chain-llm", daemon=True)
    worker.start()

    while True:
        piece = pipe.get()
        if piece is _SENTINEL:
            break
        yield piece, None

    worker.join()
    error = outcome.get("error")
    if error is not None:
        raise error
    yield None, outcome.get("result", "")


class _Gpu:
    """The workload lock, acquired with something to say while it waits."""

    def __init__(self, label: str, cancel: Cancellation, role: str = ""):
        self.label, self.cancel, self.role = label, cancel, role
        self._held = None
        self._workload = None

    def _domain(self):
        """Where the server that will answer this request executes.

        Resolved from the role's *current* configuration, every time it is
        asked, which is what makes the post-lock re-check in :meth:`acquire`
        able to see a setting the user changed while this request was queued
        (section 16.4).
        """
        try:
            import mc_llm_runtime

            return mc_llm_runtime.execution_domain(mc_llm_runtime.config(self.role))
        except Exception:
            logger.debug("Model Chain: could not resolve which device will serve this "
                         "request", exc_info=True)
            return mc_broker.UNKNOWN_CUDA_EXECUTION

    @staticmethod
    def _blocked_by_the_image_job(domain) -> bool:
        """Whether a running image generation is competing with ``domain``.

        The narrowing this whole change set is for. "The host is busy" used to
        be the whole predicate, which on a two-card machine meant a
        conversation on the 5090 waited for every 3090 generation for no reason
        anybody could have named -- the two share no processor, and the
        language model's tokens would have cost the image pass nothing.

        Still conservative where it must be: an unresolved card waits, and so
        does an image job whose card cannot be read, because being wrong that
        way costs some throughput and being wrong the other way puts two jobs
        on one card with neither expecting the other (invariants I-1 and I-10).
        """
        if not mc_broker.host_busy():
            return False
        return domain.conflicts_with(mc_broker.image_execution_domain())

    def acquire(self):
        """Generator: yields status events until this request may begin.

        Two conditions, not one. The lock keeps LLM turns from overlapping each
        other; the image check keeps a turn from starting on top of an image
        generation *it would actually contend with*, which no lock this
        extension holds could see. The second is re-checked after the lock is
        taken, because the check and the acquisition are not atomic and the
        point of checking is to be right at the moment work actually begins --
        and because a configuration changed while this request was queued must
        not start on stale scope (section 16.4).

        The second condition has one exception, and it is
        ``mc_broker.inside_host_job()``: a turn the image generation is itself
        blocked waiting for is not a turn competing with that generation. Krea
        Creative Mode's roll runs from inside ``before_process``, so waiting for
        the host to go idle there would be waiting for the job that is waiting
        for this. The lock is still taken -- two LLM turns still serialise --
        and that wait terminates, because the turn ahead is running rather than
        waiting on anything the image job holds.
        """
        started = time.monotonic()
        announced = False
        while True:
            if self.cancel.is_set():
                return False
            ours = mc_broker.inside_host_job()
            domain = self._domain()
            if not ours and self._blocked_by_the_image_job(domain):
                # Named only when the name adds something. On the ordinary
                # single-card machine "image generation" is the whole truth and
                # "image generation on an unidentified GPU" is alarming prose
                # about a card that was never in doubt; on a two-card machine
                # the card is exactly what somebody wants to know, because it
                # is why this request is waiting at all.
                where = mc_broker.image_execution_domain()
                announced = yield from self._announce(
                    started, announced,
                    f"image generation on {where.describe()}" if where.known
                    else "image generation")
                time.sleep(WAIT_POLL_SECONDS)
                continue
            self._workload = mc_broker.workload(mc_broker.FAMILY_LLM, self.label,
                                                timeout=WAIT_POLL_SECONDS, required=False,
                                                domain=domain)
            self._held = self._workload.__enter__()
            if self._held:
                # Re-resolved rather than reused. A request queued for GPU 1
                # that the user moved to GPU 0 while it waited must not start
                # on top of the generation running there, and the recorded
                # domain must describe where this turn really executes --
                # everything on the image side reads it to decide whether to
                # wait for this one.
                again = self._domain()
                if again != domain:
                    self.release()
                    continue
                if ours or not self._blocked_by_the_image_job(again):
                    return True
                self.release()
                continue
            self._workload.__exit__(None, None, None)
            self._workload = None
            running = mc_broker.active()
            if not announced and time.monotonic() - started > WAIT_NOTICE_SECONDS:
                # Logged as well as shown, and with the holder's age, because
                # the two things this wait can mean look identical on screen: a
                # machine that is genuinely busy, and a lease that outlived the
                # run that took it. An age longer than any reply could plausibly
                # take is the difference, and without it every report of
                # "Waiting for a conversation reply…" costs a round of guessing.
                logger.info(
                    "Model Chain: %s is waiting for the GPU — held by %s for %.0f s",
                    self.label,
                    running.describe() if running else "nothing this module can see",
                    0.0 if running is None else max(0.0, time.monotonic() - running.since))
            announced = yield from self._announce(
                started, announced, running.label if running else "the GPU")

    def _announce(self, started: float, announced: bool, what: str):
        """Say what is being waited for -- once, and only once the wait is real.

        Returns the new "have we said something" flag, which is why it is
        consumed with ``yield from`` rather than iterated: a wait that resolves
        inside the notice window should say nothing at all, and a wait that
        outlasts it should say it once rather than four times a second.
        """
        if announced or time.monotonic() - started <= WAIT_NOTICE_SECONDS:
            return announced
        yield Event(STATUS, f"Waiting for {what}…")
        return True

    def release(self):
        """Give the card back. Idempotent, and called *before* the last word.

        The ``finally`` in every generator below is a backstop, not the
        mechanism, and the difference is what this docstring is for.

        A generator's ``finally`` runs when the generator is resumed to
        completion or closed. Gradio's ``cancels=`` does neither reliably: a
        run stopped while it was streaming has already yielded its terminal
        event, the panel returns out of its own ``for`` loop, and the handler
        generator is left in the queue's hands. Until something finalises it
        the card is still booked to a reply that finished, and every later run
        on another thread queues behind it -- "Waiting for a conversation
        reply…", from a conversation reply that is over. Nothing recovers it,
        because :class:`mc_broker._JobLock` is reentrant by thread identity, so
        the only runs that get through are the ones that happen to land on the
        thread holding the leak.

        That was visible in one user's log as three runs that started and never
        reported an end at all, and as every stop-then-retry pair after the
        first stop mid-stream. Runs stopped while *waiting* were fine, because
        they held nothing to lose.

        So the release happens on the way out, before the terminal event is
        yielded, and finalisation is left to tidy up rather than to matter. By
        the time the panel sees Stopped, Complete or an error, the GPU is
        already somebody else's to take.
        """
        if self._workload is not None:
            self._workload.__exit__(None, None, None)
            self._workload = None
            self._held = None


def _client(needs_vision: bool, reserve: int = 0, role: str = "", cancel=None):
    """A ready client, optionally promising to leave ``reserve`` bytes of VRAM.

    Only Krea's Creative Mode passes anything: it is the one caller that knows
    another workload -- an image generation on a checkpoint several gigabytes
    wide -- starts moments after the writer finishes. Every other mode leaves
    it at zero and behaves exactly as it did.

    ``role`` picks which configured LLM answers. Empty -- which is what every
    mode but the Krea writer and the Composer passes -- is the installation's
    own, exactly as before roles existed. A role that has not been configured
    separately resolves to that same installation and gets the same running
    server back, so this parameter costs nothing until somebody uses it.

    ``cancel`` is this run's stop button, and it is passed on for one reason:
    an image request against a managed backbone whose projector has gone
    missing repairs it first, and a repair is a download. Without this the Stop
    button would appear to do nothing until the transfer finished. What arrives
    is kept either way, so a cancelled repair resumes rather than restarts.
    """
    chosen = _runtime_for(role)
    if role:
        try:
            chosen.registry.make_room_for(role, chosen.configuration)
        except Exception:
            logger.debug("Model Chain: could not check the other role's runtime",
                         exc_info=True)
    return chosen.runtime.client(needs_vision, reserve=reserve, cancel=cancel)


class _Resolved(NamedTuple):
    """One role's runtime and the configuration it was resolved from."""

    runtime: object
    configuration: object
    registry: object


def _runtime_for(role: str = "") -> _Resolved:
    """Which server serves ``role``, and what it was configured from.

    One lookup rather than four, because the status line, the placement notes
    and the speed reading all have to be about the *same* server as the request
    -- and with two of them up, asking the module singleton is how a Composer
    running on the processor comes to report the writer's card.
    """
    registry = mc_llm_runtime.registry
    configuration = mc_llm_runtime.config(role)
    return _Resolved(registry.for_role(role, configuration), configuration, registry)


def _preparing(role: str = "") -> str:
    """What to say before asking for a client, which may not start anything.

    Every mode used to announce "Starting llama-server…" before every request,
    which was true when every request restarted the server and is a lie now
    that a warm one is handed back untouched. The distinction is worth a
    branch rather than a vaguer sentence that covers both: a cold start is
    twenty seconds of reading weights and the reason the tab looks stuck, and
    somebody watching it deserves to know which of the two they are waiting
    for.
    """
    try:
        if _runtime_for(role).runtime.running():
            return "Preparing…"
    except Exception:  # a runtime that cannot say is a runtime about to start
        logger.debug("Model Chain: could not ask the LLM runtime whether it is up",
                     exc_info=True)
    return "Starting llama-server…"


def _measured(role: str = "") -> str:
    """llama.cpp's own timing for the reply that just finished, if it said one.

    Characters per second is what this module can count and is not what anybody
    means by speed: it depends on the tokenizer, the language and how chatty
    the model was feeling. The server measures the real thing per request and
    writes it to its log, and a line that carries both is a line somebody can
    compare across two placements without arithmetic.
    """
    try:
        note = _runtime_for(role).runtime.speed_note()
    except Exception:
        logger.debug("Model Chain: could not read llama.cpp's timings", exc_info=True)
        return ""
    return f" ({note})" if note else ""


def _placement_notes(role: str = "") -> list[Event]:
    """Report anything negotiation changed (section 13)."""
    report = _runtime_for(role).runtime.report
    return [Event(STATUS, note) for note in (report.notes or ())]


# --------------------------------------------------------------------------- #
# Prompt Studio (section 4.2)
# --------------------------------------------------------------------------- #


def _prompt_studio(request, cancel: Cancellation):
    """One LTX prompt generation.

    The same order the standalone worker used, and the order matters at two
    points that are easy to get wrong:

    * speech expansion runs *before* the brief is built, because the extra
      spoken lines have to be in the intent the engine reads rather than bolted
      onto the shot it wrote;
    * the smart-negative pass runs *after* the positive is finished, because it
      is written about the script that came out.
    """
    from prompt_master.prompt_engine import speech
    from prompt_master.prompt_engine.adapter import PromptEngine

    engine = PromptEngine()
    gpu = _Gpu("an LTX prompt", cancel)
    try:
        acquired = yield from gpu.acquire()
        if not acquired:
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return

        needs_vision = bool(request.image_data_url) and request.video_mode == "i2v"
        yield Event(STATUS, _preparing())
        client = _client(needs_vision, cancel=cancel.event)
        for event in _placement_notes():
            yield event

        if request.speech > speech.NONE:
            yield Event(STATUS, "Writing extra speech…")
            request, note = speech.expand(request, _chat_stream(client, request.seed, cancel),
                                          seed=request.seed)
            if note:
                yield Event(STATUS, f"Intent expanded — {note}")
            if cancel.is_set():
                gpu.release()
                yield Event(CANCELLED, "Cancelled")
                return

        plan = engine.build(request, vision_available=True)
        yield Event(NEGATIVE, engine.base_negative(request))
        yield Event(STATUS,
                    f"Generating positive prompt… ({plan.frames} frames, "
                    f"{plan.word_budget[0]}-{plan.word_budget[1]} words)")

        temperature, top_p = engine.sampling(request)
        raw = ""
        for chunk, result in _streamed(
                lambda on_text: client.stream_chat(plan.messages, plan.max_tokens, request.seed,
                                                   on_text, cancel.event,
                                                   temperature=temperature, top_p=top_p)):
            if chunk is not None:
                yield Event(CHUNK, chunk)
            else:
                raw = result or ""

        if cancel.is_set():
            gpu.release()
            yield Event(CANCELLED, "Generation cancelled")
            return

        positive = engine.clean_positive(raw)
        if not positive.strip():
            raise RuntimeError("The model returned an empty script.")
        yield Event(POSITIVE, positive)

        auto = ""
        if request.smart_negative:
            yield Event(STATUS, "Negative pass…")
            auto = engine.run_smart_negative(positive, _chat_stream(client, request.seed, cancel))
        yield Event(NEGATIVE, engine.merge_negative(request, auto))
        gpu.release()
        yield Event(DONE, f"Complete · Seed: {request.seed}")
    except Exception as exc:
        logger.debug("Model Chain: Prompt Studio run failed", exc_info=True)
        gpu.release()
        yield Event(FAILED, str(exc))
    finally:
        gpu.release()


def _chat_stream(client, seed: int, cancel: Cancellation):
    """Shim matching upstream ``backend.chat_stream``.

    Kept so ``negative.run_auto`` and ``speech.expand`` -- their temperatures,
    their guards, their never-raises contracts -- are used exactly as vendored
    rather than reimplemented against a different calling convention. Both call
    it as ``chat_stream(messages, temperature=, top_p=, max_tokens=, seed=)``
    and join what comes back, so it returns a one-element list of the whole
    text rather than streaming: neither pass shows its output as it arrives.

    ``seed=None`` falls back to the run's own seed, which is what makes a
    generation reproducible end to end rather than only in its writer pass.
    """
    fallback = int(seed)
    # The signature's defaults are upstream ``backend.chat_stream``'s, and are
    # never the ones used: every caller in the prompt engine passes its own
    # pass's temperature. They are here so the shim is substitutable for the
    # function it stands in for, not as a policy.
    writer, writer_top_p = _writer_defaults()

    def stream(messages, *, temperature=writer, top_p=writer_top_p, max_tokens=900, seed=None):
        chosen = fallback if seed is None else int(seed)
        return [client.stream_chat(messages, max_tokens, chosen, lambda _text: None,
                                   cancel.event, temperature=temperature, top_p=top_p)]

    return stream


def _writer_defaults() -> tuple[float, float]:
    """Upstream ``backend.chat_stream``'s own sampling, from the vendored copy."""
    from prompt_master.chat import characters

    return characters.DEFAULT_TEMPERATURE, characters.DEFAULT_TOP_P


# --------------------------------------------------------------------------- #
# Conversation (section 4.3)
# --------------------------------------------------------------------------- #


def _vendored(name: str):
    """A default read from ``prompt_master.chat.characters`` when it is needed.

    A factory rather than the constant itself because a dataclass field default
    is evaluated at import, and nothing in this module may import the vendored
    package then -- section 18's rule that a failure in the LLM half must not
    reach a WebUI that never opens it. Evaluated per request is late enough.
    """
    def read():
        from prompt_master.chat import characters

        return getattr(characters, name)

    return read


@dataclass
class ChatRequest:
    """One reply, in the terms ``prompt_master.chat`` already speaks.

    The sampling defaults are the vendored package's own constants rather than
    literals: they are what the standalone application ships with, and a
    second copy of them here is how the two quietly stop agreeing.
    """

    messages: list
    needs_vision: bool = False
    temperature: float = field(default_factory=_vendored("DEFAULT_TEMPERATURE"))
    top_p: float = field(default_factory=_vendored("DEFAULT_TOP_P"))
    max_tokens: int = field(default_factory=_vendored("DEFAULT_MAX_REPLY_TOKENS"))
    seed: int = 0


def _conversation(request: ChatRequest, cancel: Cancellation):
    """One streamed reply. Conversation's whole runtime story is this function."""
    gpu = _Gpu("a conversation reply", cancel)
    try:
        acquired = yield from gpu.acquire()
        if not acquired:
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return

        yield Event(STATUS, _preparing())
        client = _client(request.needs_vision, cancel=cancel.event)
        for event in _placement_notes():
            yield event
        yield Event(STATUS, "Replying…")

        text = ""
        for chunk, result in _streamed(
                lambda on_text: client.stream_chat(request.messages, request.max_tokens,
                                                   request.seed, on_text, cancel.event,
                                                   temperature=request.temperature,
                                                   top_p=request.top_p)):
            if chunk is not None:
                yield Event(CHUNK, chunk)
            else:
                text = result or ""

        if cancel.is_set():
            # Not an error and not a discard: what was streamed before the stop
            # is a real partial reply and the caller keeps it.
            gpu.release()
            yield Event(CANCELLED, text)
            return
        gpu.release()
        yield Event(DONE, text)
    except Exception as exc:
        logger.debug("Model Chain: conversation run failed", exc_info=True)
        gpu.release()
        yield Event(FAILED, str(exc))
    finally:
        gpu.release()


# --------------------------------------------------------------------------- #
# MiniMax H3 (section 4.4)
# --------------------------------------------------------------------------- #


def _minimax(prompt: str, variant: str, image: str | None, seed: int,
             cancel: Cancellation):
    """One H3 prompt, in WanGP's order: caption first, then the prompt built from it.

    Kept a separate function from :func:`conversation` even though both stream
    from the same server, because section 4.4 asks for the enhancer to stay a
    dedicated workflow. The caption is emitted as its own event for the same
    reason -- it is a step of this product, not a chat turn.
    """
    from prompt_master.minimax import enhancer

    gpu = _Gpu("a MiniMax prompt", cancel)
    try:
        acquired = yield from gpu.acquire()
        if not acquired:
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return

        yield Event(STATUS, _preparing())
        client = _client(image is not None, cancel=cancel.event)
        for event in _placement_notes():
            yield event

        caption = None
        if image is not None:
            yield Event(STATUS, "Describing the image…")
            described = client.stream_chat(
                enhancer.caption_messages(image), enhancer.CAPTION_MAX_TOKENS, seed,
                lambda _text: None, cancel.event,
                temperature=enhancer.CAPTION_TEMPERATURE, top_p=enhancer.CAPTION_TOP_P)
            if cancel.is_set():
                gpu.release()
                yield Event(CANCELLED, "Cancelled")
                return
            caption = enhancer.clean(described)
            if not caption:
                raise RuntimeError("The model returned no description of the image.")
            yield Event(CAPTION, caption)

        yield Event(STATUS, f"{enhancer.label(variant, caption is not None)}…")
        written = ""
        for chunk, result in _streamed(
                lambda on_text: client.stream_chat(
                    enhancer.messages(prompt, variant=variant, image_caption=caption),
                    enhancer.max_tokens(variant), seed, on_text, cancel.event,
                    temperature=enhancer.TEMPERATURE, top_p=enhancer.TOP_P)):
            if chunk is not None:
                yield Event(CHUNK, chunk)
            else:
                written = result or ""

        if cancel.is_set():
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return
        cleaned = enhancer.clean(written)
        if not cleaned:
            raise RuntimeError("The model returned an empty prompt.")
        gpu.release()
        yield Event(DONE, cleaned)
    except Exception as exc:
        logger.debug("Model Chain: MiniMax run failed", exc_info=True)
        gpu.release()
        yield Event(FAILED, str(exc))
    finally:
        gpu.release()


# --------------------------------------------------------------------------- #
# Krea 2
# --------------------------------------------------------------------------- #


def _krea(prompt: str, references, seed: int, cancel: Cancellation, creativity=None,
          direction: str = "", reserve: int = 0):
    """One Krea 2 prompt: every reference described in order, then the writing.

    The same caption-first shape as :func:`_minimax` and for the same reason --
    the writer is a text pass over descriptions, so no multi-image transport is
    needed and every description is a thing that can be shown and kept. What is
    different is that there may be up to four of them, and that their order is
    load-bearing: "the woman from image 2" is only meaningful if image 2 is the
    second slot the user filled and stays the second caption the writer reads.

    So the loop is sequential rather than concurrent. Not for want of a thread
    pool -- one server answers one request at a time here anyway -- but because
    a sequential loop emits its captions in slot order by construction, and the
    panel can pair the first CAPTION event with Image 1 without either side
    carrying an index.

    Two failures are deliberately fatal rather than survivable. A reference that
    cannot be described stops the run, because writing the prompt from the other
    three would either renumber them behind the user's back or write about an
    image that is not there. And an empty finished prompt is an error, not an
    empty box.

    ``creativity`` is the 0-10 Creativity position, resolved to sampling settings
    by :mod:`prompt_master.krea.variation`. It reaches the *writer* and nothing
    else. The captioner keeps temperature 0 at every position, because a
    description of what is in a photograph has nothing for a sampler to be
    creative about, and a creative captioner is a captioner writing about a
    picture that is not there.

    ``direction`` is Creative Mode's finished brief, assembled locally by
    :mod:`prompt_master.krea.director` before this generator was called. It is a
    string by the time it arrives, and that is the invariant worth stating: no
    model chose it, nothing here can add to it, and there is no branch below
    that could turn it into a second request. An empty string is the whole of
    what "Creative Mode off" means down here.

    ``reserve`` is VRAM the writer promises to leave alone because an image
    generation follows it. It reaches :func:`_client` and nothing else, and it
    is zero for every caller that is not about to make a picture.

    Exactly one writer request is made, at every Creativity value and with or
    without a brief. The captions are one request per reference and are
    reference *processing*, not prompt writing; nothing here multiplies either
    count by the slider.
    """
    from prompt_master.krea import enhancer
    from prompt_master.krea.variation import creativity_profile, resolve

    references = list(references or [])
    profile = creativity_profile(resolve(creativity))
    gpu = _Gpu("a Krea prompt", cancel, mc_llm_roles.CREATIVE)
    try:
        acquired = yield from gpu.acquire()
        if not acquired:
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return

        yield Event(STATUS, _preparing(mc_llm_roles.CREATIVE))
        # Vision is asked for only when there is something to look at, which is
        # what lets a text-only Krea prompt be written by any model that can
        # hold a conversation. The panel has already refused a run that would
        # need a projector the model has not got; this is the same requirement
        # stated where the client is actually obtained.
        client = _client(bool(references), reserve, mc_llm_roles.CREATIVE,
                         cancel=cancel.event)
        for event in _placement_notes(mc_llm_roles.CREATIVE):
            yield event

        captions: list[str] = []
        for position, reference in enumerate(references, start=1):
            yield Event(STATUS, enhancer.caption_label(position, len(references)))
            described = client.stream_chat(
                enhancer.caption_messages(reference.data_url), enhancer.CAPTION_MAX_TOKENS,
                seed, lambda _text: None, cancel.event,
                temperature=enhancer.CAPTION_TEMPERATURE, top_p=enhancer.CAPTION_TOP_P)
            if cancel.is_set():
                gpu.release()
                yield Event(CANCELLED, "Cancelled")
                return
            caption = enhancer.clean(described)
            if not caption:
                raise RuntimeError(
                    f"The model returned no description of image {position}. "
                    "The prompt has not been written: the other references would "
                    "have to be renumbered to write it without this one.")
            reference.caption = caption
            captions.append(caption)
            yield Event(CAPTION, caption)

        yield Event(STATUS, f"{enhancer.label(len(references))}…")
        written = ""
        for chunk, result in _streamed(
                lambda on_text: client.stream_chat(
                    enhancer.messages(prompt, captions, direction), enhancer.MAX_TOKENS,
                    seed, on_text, cancel.event,
                    temperature=profile.temperature, top_p=profile.top_p)):
            if chunk is not None:
                yield Event(CHUNK, chunk)
            else:
                written = result or ""

        if cancel.is_set():
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return
        cleaned = enhancer.clean(written)
        if not cleaned:
            raise RuntimeError("The model returned an empty prompt.")
        gpu.release()
        yield Event(DONE, cleaned)
    except Exception as exc:
        logger.debug("Model Chain: Krea run failed", exc_info=True)
        gpu.release()
        yield Event(FAILED, str(exc))
    finally:
        gpu.release()


def _compose(source: str, scene: str, layout, ratio: str, seed: int,
             cancel: Cancellation, reserve: int = 0):
    """One Spatial Composer pass: two strings back, and nothing else read.

    The second language-model request Krea Creative Mode can make, and the only
    one that is optional. It runs when the user chose Smart Spatial Compose *and*
    drew at least one usable region; Direct BBOX Merge makes no request at all
    and is one radio button away, which is what keeps this an opt-in cost rather
    than a tax on Creative Mode.

    Three things about it are deliberately unlike :func:`_krea` and each of them
    is a decision rather than an omission:

    *No references, ever.* This pass edits two paragraphs. It is asked for a
    text-only client, so it can run on any backbone that can hold a conversation
    and never pulls a vision projector onto the card for a job with no picture
    in it.

    *Its own instruction, not Krea's.* Krea's expansion instruction says expand,
    and says no JSON. Both are the opposite of what this pass does. See
    :mod:`prompt_master.krea.composer` for what that costs in llama.cpp's prompt
    cache and why it is still the right way round.

    *Failure is not fatal.* Every exit that is not a finished, validated pair of
    strings is a ``FAILED`` event, and the caller's answer to that is Direct
    merge -- the first pass's scene, the same boxes, no second request. The image
    is never cancelled because a copy-editor was unavailable.

    ``reserve`` is the same promise :func:`_krea` makes: VRAM this pass will
    leave alone because an image generation follows it.
    """
    from prompt_master.krea import composer

    gpu = _Gpu("the spatial composition", cancel, mc_llm_roles.SPATIAL)
    try:
        acquired = yield from gpu.acquire()
        if not acquired:
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return

        yield Event(STATUS, _preparing(mc_llm_roles.SPATIAL))
        client = _client(False, reserve, mc_llm_roles.SPATIAL)
        for event in _placement_notes(mc_llm_roles.SPATIAL):
            yield event

        yield Event(STATUS, COMPOSING)
        written = ""
        for chunk, result in _streamed(
                lambda on_text: client.stream_chat(
                    composer.messages(source, scene, layout, ratio),
                    composer.MAX_TOKENS, seed, on_text, cancel.event,
                    temperature=composer.TEMPERATURE, top_p=composer.TOP_P)):
            if chunk is not None:
                yield Event(CHUNK, chunk)
            else:
                written = result or ""

        if cancel.is_set():
            gpu.release()
            yield Event(CANCELLED, "Cancelled")
            return

        # Counted, not acted on. parse() has already ignored anything that is
        # not one of the two strings; this is the only place a Composer that
        # keeps trying to write the elements array becomes visible.
        reached = composer.overreached(written)
        if reached:
            logger.warning("Model Chain: the Spatial Composer returned %s, which was "
                           "ignored — the layout is the user's",
                           ", ".join(reached))
        composed, background = composer.parse(written)
        gpu.release()
        yield Event(DONE, composed, data={"scene": composed, "background": background})
    except Exception as exc:
        logger.debug("Model Chain: the Spatial Composer failed", exc_info=True)
        gpu.release()
        yield Event(FAILED, str(exc))
    finally:
        gpu.release()


COMPOSING = "Reconciling the scene with the layout"
"""What the Composer pass is doing, for the status line and the progress bar."""


# --------------------------------------------------------------------------- #
# What the console is told
# --------------------------------------------------------------------------- #

PROGRESS_SECONDS = 5.0
"""How often a run in progress says so on the console.

A line per chunk would be a line per token. This is slow enough to read in a
terminal that is also carrying a generation's progress bar, and quick enough
that a run which has stalled is visibly not moving.

The clock these lines run on starts at the *first token*, not at the request.
It used to start at the request, and every line then folded two different things
into one rate: llama.cpp reads the whole prompt before it emits anything, so the
first line of a pass that had 230 new tokens to read said "generating, 3
characters in 5s". Nothing was generating slowly. Nothing was generating at all
for the first 4.6 seconds of that, and a reader with no way to tell the two
apart reasonably concluded the server was warming up. See :func:`_traced`.
"""


def _role_in(label: str) -> str:
    """Which role a traced label belongs to, read back off its own prefix.

    A small indignity, and the alternative is threading the role through every
    event a generator yields so that one timing line can be read off the right
    server. The prefix is written by exactly two callers a dozen lines apart and
    is the only thing in a label that ever looks like this.
    """
    for role in mc_llm_roles.ROLES:
        if label.startswith(mc_llm_roles.prefix(role)):
            return role
    return mc_llm_roles.SHARED


def _traced(label: str, events):
    """Pass ``events`` through, and narrate the run to the WebUI console.

    Every mode is wrapped in this rather than each of them logging for itself,
    because what is worth saying is the same for all three: it started, what it
    is waiting for or doing, how it ended and how long it took. The three
    generators below stay about their own products.

    **Nothing written here is content.** The prompt, the reply, the message, the
    character and the thread are never logged — a status line saying a reply is
    2,300 characters long is a status line; one quoting it is a transcript of a
    private conversation in a file somebody else may read. What is logged is the
    kind of run, what stage it reached, sizes, and elapsed time. Failures are
    logged with their message, because the whole purpose of that message is to
    be read by the person who has to fix it, and the vendored layers raise
    sentences about paths and servers rather than about what was said.

    A ``yield from`` and not a loop, so that closing the outer generator —
    which is what Gradio's ``cancels=`` does — still closes the inner one and
    still runs the ``finally`` that gives the GPU back.

    "Abandoned" is only said about a run that was actually abandoned. Gradio
    closes a handler's generator once it has consumed the last event, so a run
    that finished normally raises ``GeneratorExit`` here a moment after saying
    it was complete — and a console that reported both read as though every
    reply had gone wrong at the end. What is worth a line is the other case:
    the tab was closed, or the queue dropped the request, while the run still
    had work to do.
    """
    started = time.monotonic()
    streamed = 0
    spoke_at = started
    writing_from = None
    """When the first token arrived, which is when generating actually began."""
    finished = False
    logger.info("Model Chain: LLM run started — %s", label)
    try:
        for event in events:
            if event.kind == CHUNK:
                streamed += len(event.text or "")
                now = time.monotonic()
                if writing_from is None:
                    writing_from, spoke_at = now, now
                    logger.info("Model Chain: LLM %s — prompt read in %.1fs, writing now",
                                label, now - started)
                elif now - spoke_at >= PROGRESS_SECONDS:
                    spoke_at = now
                    logger.info("Model Chain: LLM %s — generating, %s characters in %.0fs",
                                label, f"{streamed:,}", now - writing_from)
            elif event.kind == STATUS:
                logger.info("Model Chain: LLM %s — %s", label, event.text)
            elif event.kind == DONE:
                finished = True
                logger.info("Model Chain: LLM run finished — %s, %s characters in %.1fs%s%s",
                            label, f"{max(streamed, len(event.text or '')):,}",
                            time.monotonic() - started,
                            "" if writing_from is None
                            else f" ({writing_from - started:.1f}s reading, "
                                 f"{time.monotonic() - writing_from:.1f}s writing)",
                            _measured(_role_in(label)))
            elif event.kind == CANCELLED:
                finished = True
                logger.info("Model Chain: LLM run cancelled — %s after %.1fs",
                            label, time.monotonic() - started)
            elif event.kind == FAILED:
                finished = True
                logger.warning("Model Chain: LLM run failed — %s after %.1fs: %s",
                               label, time.monotonic() - started, event.text)
            yield event
    except GeneratorExit:
        if not finished:
            logger.info("Model Chain: LLM run abandoned — %s after %.1fs",
                        label, time.monotonic() - started)
        raise


def prompt_studio(request, cancel: Cancellation):
    """One LTX prompt generation. See :func:`_prompt_studio`."""
    yield from _traced("a prompt generation", _prompt_studio(request, cancel))


def conversation(request: ChatRequest, cancel: Cancellation):
    """One streamed conversation reply. See :func:`_conversation`."""
    yield from _traced("a conversation reply", _conversation(request, cancel))


def minimax(prompt: str, variant: str, image: str | None, seed: int, cancel: Cancellation):
    """One MiniMax enhancement. See :func:`_minimax`."""
    yield from _traced(f"a MiniMax {variant} enhancement",
                       _minimax(prompt, variant, image, seed, cancel))


def krea(prompt: str, references, seed: int, cancel: Cancellation, creativity=None,
         direction: str = "", reserve: int = 0):
    """One Krea 2 prompt. See :func:`_krea`.

    ``references`` is an ordered list of ``prompt_master.krea.references
    .Reference`` -- empty for a text-only prompt. The count reaches the console
    line; the pictures, the paths and the captions do not.

    ``creativity`` reaches the console line as a number, and whether a creative
    brief was attached reaches it as a word. Both are settings rather than
    content, and a run that came back unlike its neighbours is a run somebody
    will want those two facts about. The brief's *text* is not logged: it names
    what the user asked for.
    """
    from prompt_master.krea.variation import resolve

    references = list(references or [])
    counted = (f"a Krea prompt from {len(references)} reference"
               f"{'' if len(references) == 1 else 's'}" if references else "a Krea prompt")
    directed = " with creative direction" if str(direction or "").strip() else ""
    yield from _traced(f"{mc_llm_roles.prefix(mc_llm_roles.CREATIVE)}{counted} at "
                       f"creativity {resolve(creativity)}{directed}",
                       _krea(prompt, references, seed, cancel, creativity, direction,
                             reserve))


def krea_compose(source: str, scene: str, layout, ratio: str, seed: int,
                 cancel: Cancellation, reserve: int = 0):
    """One Spatial Composer pass. See :func:`_compose`.

    The region count reaches the console line because a run that took twice as
    long as its neighbour is a run somebody will want that number about. The
    region prompts, the visible text and the coordinates do not: they are the
    user's own words about their own picture, and the rule this module already
    follows is that a status line says what kind of run it was and never what
    was in it.
    """
    counted = len(getattr(layout, "regions", ()) or ())
    yield from _traced(f"{mc_llm_roles.prefix(mc_llm_roles.SPATIAL)}a spatial composition "
                       f"over {counted} region{'' if counted == 1 else 's'}",
                       _compose(source, scene, layout, ratio, seed, cancel, reserve))
