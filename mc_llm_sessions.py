"""The three LLM modes, as streaming generators with no UI in them.

Section 4 requires Prompt Studio, Conversation and MiniMax to stay distinct
products rather than collapsing into one chat workflow, and section 17.B
requires the LTX business logic to come across without its Qt presentation
layer. Both are served by the same decision: the orchestration each mode's Qt
worker used to do lives here, in a form that has never heard of a widget.

Each mode is a generator of :class:`Event`. The Qt version pushed the same
information out as signals -- ``status``, ``chunk``, ``positive_ready``,
``negative_ready``, ``captioned``, ``failed`` -- and the sequence of events
below is deliberately the same sequence, because that sequence *is* the
product. Prompt Studio still emits a positive and a negative separately;
MiniMax still emits its caption before its prompt; Conversation still emits one
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
from dataclasses import dataclass

import mc_broker
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

    def __init__(self, label: str, cancel: Cancellation):
        self.label, self.cancel = label, cancel
        self._held = None
        self._workload = None

    def acquire(self):
        """Generator: yields status events until the GPU is ours.

        Two conditions, not one. The lock keeps LLM turns from overlapping each
        other; ``host_busy`` keeps a turn from starting on top of an image
        generation the host is running, which no lock this extension holds
        could see. The second is re-checked after the lock is taken, because
        the check and the acquisition are not atomic and the point of checking
        is to be right at the moment work actually begins.
        """
        started = time.monotonic()
        announced = False
        while True:
            if self.cancel.is_set():
                return False
            if mc_broker.host_busy():
                announced = yield from self._announce(started, announced, "image generation")
                time.sleep(WAIT_POLL_SECONDS)
                continue
            self._workload = mc_broker.workload(mc_broker.FAMILY_LLM, self.label,
                                                timeout=WAIT_POLL_SECONDS, required=False)
            self._held = self._workload.__enter__()
            if self._held:
                if not mc_broker.host_busy():
                    return True
                self.release()
                continue
            self._workload.__exit__(None, None, None)
            self._workload = None
            running = mc_broker.active()
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
        if self._workload is not None:
            self._workload.__exit__(None, None, None)
            self._workload = None
            self._held = None


def _client(needs_vision: bool):
    return mc_llm_runtime.runtime.client(needs_vision)


def _placement_notes() -> list[Event]:
    """Report anything negotiation changed (section 13)."""
    report = mc_llm_runtime.runtime.report
    return [Event(STATUS, note) for note in (report.notes or ())]


# --------------------------------------------------------------------------- #
# Prompt Studio (section 4.2)
# --------------------------------------------------------------------------- #


def prompt_studio(request, cancel: Cancellation):
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
            yield Event(CANCELLED, "Cancelled")
            return

        needs_vision = bool(request.image_data_url) and request.video_mode == "i2v"
        yield Event(STATUS, "Starting llama-server…")
        client = _client(needs_vision)
        for event in _placement_notes():
            yield event

        if request.speech > speech.NONE:
            yield Event(STATUS, "Writing extra speech…")
            request, note = speech.expand(request, _chat_stream(client, request.seed, cancel),
                                          seed=request.seed)
            if note:
                yield Event(STATUS, f"Intent expanded — {note}")
            if cancel.is_set():
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
        yield Event(DONE, f"Complete · Seed: {request.seed}")
    except Exception as exc:
        logger.debug("Model Chain: Prompt Studio run failed", exc_info=True)
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

    def stream(messages, *, temperature=0.85, top_p=0.95, max_tokens=900, seed=None):
        chosen = fallback if seed is None else int(seed)
        return [client.stream_chat(messages, max_tokens, chosen, lambda _text: None,
                                   cancel.event, temperature=temperature, top_p=top_p)]

    return stream


# --------------------------------------------------------------------------- #
# Conversation (section 4.3)
# --------------------------------------------------------------------------- #


@dataclass
class ChatRequest:
    """One reply, in the terms ``prompt_master.chat`` already speaks."""

    messages: list
    needs_vision: bool = False
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 512
    seed: int = 0


def conversation(request: ChatRequest, cancel: Cancellation):
    """One streamed reply. Conversation's whole runtime story is this function."""
    gpu = _Gpu("a conversation reply", cancel)
    try:
        acquired = yield from gpu.acquire()
        if not acquired:
            yield Event(CANCELLED, "Cancelled")
            return

        yield Event(STATUS, "Starting llama-server…")
        client = _client(request.needs_vision)
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
            yield Event(CANCELLED, text)
            return
        yield Event(DONE, text)
    except Exception as exc:
        logger.debug("Model Chain: conversation run failed", exc_info=True)
        yield Event(FAILED, str(exc))
    finally:
        gpu.release()


# --------------------------------------------------------------------------- #
# MiniMax H3 (section 4.4)
# --------------------------------------------------------------------------- #


def minimax(prompt: str, variant: str, image: str | None, seed: int, cancel: Cancellation):
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
            yield Event(CANCELLED, "Cancelled")
            return

        yield Event(STATUS, "Starting llama-server…")
        client = _client(image is not None)
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
            yield Event(CANCELLED, "Cancelled")
            return
        cleaned = enhancer.clean(written)
        if not cleaned:
            raise RuntimeError("The model returned an empty prompt.")
        yield Event(DONE, cleaned)
    except Exception as exc:
        logger.debug("Model Chain: MiniMax run failed", exc_info=True)
        yield Event(FAILED, str(exc))
    finally:
        gpu.release()
