"""Compacting a thread: the model summarizes its older messages, and the summary
is sent in their place.

"I would like a compacting option ... instead of keeping just the latest, the
LLM can summarize the previous content into a smaller compact form that is
hidden, and slot that hidden content to be read instead of the whole window.
Maybe make it so I can choose how much to compact. Both of these option should
not cut off a message" (2026-11-09).

**What is summarized.** What the model is sent now before the latest
``keep_tokens`` of messages: the window's older messages, between whole
messages and never into the last exchange (:func:`mc_llm_window.keep_latest`),
and the thread's earlier summary with them, so the new one covers everything
the old one did. Messages the window had already left behind before this are
not read back: they were out of the model's sight already, and reading a long
thread's whole past on the Intel Arc would be minutes.

**How.** One request, in the thread's own frame: its system prompt as every
message sends it (the earlier summary and the emotion in it), then the messages
to summarize, then an instruction of the application's as a message of yours
(:data:`INSTRUCTION`). That is the beginning of the prompt llama-server has
cached for the thread, so it reads only the instruction before it writes. Cool
sampling (:data:`TEMPERATURE`), none of the thread's own: a summary is not the
place for variety. The summary's length is asked for in words, about
``summary_tokens`` of them as tokens (:data:`WORDS_PER_TOKEN`).

**Where it goes.** :func:`mc_llm_window.set_summary`: kept beside the thread,
never in its file and never shown in the transcript, sent after the system
prompt (``mc_llm_window.SUMMARY_HEAD``) with the window starting at the first
message it does not cover. The thread is then read again in the background
under its new front (``mc_llm_conversation_ops.rewarm``), so the next message
reads only itself.

**While it runs** a message in the thread is refused with a sentence
(``mc_llm_conversation_ops.begin``), the progress is on the conversation feed
(``context`` events) and in every reading of the thread, and it can be
cancelled. A thread changed under it -- a message it was summarizing edited or
deleted -- keeps nothing: a summary of words that are no longer there is not
kept.

The console lines say how many messages and tokens, and how long; never a word
of the conversation or the summary.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")

LABEL = "the compaction"
"""The language model's lock is held under this name, so anything that waits
for it is told *Waiting for the compaction…*."""

TEMPERATURE = 0.3
TOP_P = 0.9

SUMMARY_MIN = 100
SUMMARY_MAX = 1500
SUMMARY_DEFAULT = 300
"""The summary's length, in tokens: the range offered, and where it starts."""

WORDS_PER_TOKEN = 0.75

SLACK = 1.5
"""A summary may run this much past the length asked for before it is cut."""

PROGRESS_EVERY = 1.0
"""Seconds between progress events while the summary is written."""

KEPT_OUTCOME = 300.0
"""Seconds a finished compaction's outcome is answered for, so a page that asks
a little late is still told how it ended."""

INSTRUCTION = (
    "[A note from the application, not part of the conversation. Do not reply as "
    "{character}.] Write a summary of the conversation between {character} and {you} "
    "above{earlier}. It will replace those messages in your memory of this "
    "conversation, so keep everything needed to carry it on: what happened and in "
    "what order; names, places and things; what each of them said they want, feel or "
    "will do; how they stand with each other now; and anything left unresolved. "
    "Write about {words} words of plain prose in the past tense, as a narrator, not "
    "as {character}. Write only the summary: no title, no preamble and no comment "
    "afterwards."
)

EARLIER = (", folding in the summary of what came before it that is in your instructions, "
           "so that yours covers the whole conversation from its beginning")


class Refused(ValueError):
    """A compaction that cannot be started, in a sentence for the screen."""


@dataclass
class Job:
    """One thread's compaction while it runs."""

    name: str
    cancel: object
    target: int
    messages: int
    tokens: int
    cut: int
    digest: str
    phase: str = "waiting"
    status: str = "Waiting…"
    written: int = 0
    started: float = field(default_factory=time.monotonic)
    worker: object = None

    def describe(self) -> dict:
        return {"phase": self.phase, "status": self.status, "written": self.written,
                "target": self.target, "messages": self.messages, "tokens": self.tokens,
                "elapsed": round(time.monotonic() - self.started, 1)}


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_outcomes: dict[str, tuple] = {}


def _name(key) -> str:
    return f"{key.character}/{key.thread_id}"


def running(key) -> bool:
    with _lock:
        return _name(key) in _jobs


def progress(key):
    """The running compaction's progress, or how the last one ended while that is
    fresh, or None."""
    name = _name(key)
    with _lock:
        job = _jobs.get(name)
        if job is not None:
            return job.describe()
        ended = _outcomes.get(name)
    if ended and time.monotonic() - ended[2] < KEPT_OUTCOME:
        return {"phase": ended[0], "status": ended[1]}
    return None


def cancel(key) -> bool:
    """Stop this thread's compaction. Harmless when there is none."""
    with _lock:
        job = _jobs.get(_name(key))
    if job is None:
        return False
    job.cancel.cancel()
    return True


RESET_WAIT_SECONDS = 5.0
"""How long :func:`reset` waits for each stopped compaction to end."""


def reset() -> None:
    """Forget every compaction, once each has been stopped and has ended. For the
    tests: a compaction still running when its test has gone keeps its summary in
    whatever data folder is current by then -- the checkout's, once the test's
    own folder has been put back (a full run of 2026-11-10 found a window.json
    there)."""
    with _lock:
        running = list(_jobs.values())
    for job in running:
        job.cancel.cancel()
    for job in running:
        worker = job.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(RESET_WAIT_SECONDS)
    with _lock:
        _jobs.clear()
        _outcomes.clear()


def _digest(history, upto: int) -> str:
    """The words a summary of ``history[:upto]`` was written from."""
    found = hashlib.sha1()
    for message in history[:upto]:
        found.update(f"{message.role}\0{message.text}\0".encode("utf-8", "replace"))
    return found.hexdigest()


def instruction(character, persona, words: int, earlier: bool) -> str:
    name = character.name.strip() or "the character"
    return INSTRUCTION.format(character=name, you=persona.display, words=int(words),
                              earlier=EARLIER if earlier else "")


def _wire(character, persona, extra: str, span, asked: str) -> list:
    """The thread's own frame for the messages to summarize: its system prompt as
    every message sends it, the messages as the window sends them, then the
    instruction. A picture is not sent: the summary is of what was said."""
    from prompt_master.chat.prompt import substitute, system_message

    name, you = character.name.strip() or "the character", persona.display
    wire = [{"role": "system", "content": system_message(character, persona, extra)}]
    for message in span:
        text = substitute(message.text.strip(), name, you)
        if text:
            wire.append({"role": message.role, "content": text})
    wire.append({"role": "user", "content": asked})
    return wire


_HEADING = re.compile(r"^\s*(?:#+\s*|\*\*)?(?:summary|recap)\b[^\n]{0,40}\n+", re.IGNORECASE)


def clean(text: str) -> str:
    """The summary as kept: without a heading the model put on it anyway."""
    kept = (text or "").strip()
    kept = _HEADING.sub("", kept, count=1).strip()
    return kept


def start(key, keep_tokens: int, summary_tokens: int) -> dict:
    """Compact this thread, keeping the latest ``keep_tokens`` of messages as they
    are and asking for a summary of about ``summary_tokens``. Returns at once;
    the summary is written on a thread of its own. Raises :class:`Refused`."""
    import mc_llm_conversation_ops as ops
    import mc_llm_conversation_service as service
    import mc_llm_conversation_store as store_module
    import mc_llm_sessions as sessions
    import mc_llm_tokens
    import mc_llm_window

    if ops.for_conversation(key) is not None:
        raise Refused("A reply is being written. Compact when it is done.")
    target = min(max(int(summary_tokens or SUMMARY_DEFAULT), SUMMARY_MIN), SUMMARY_MAX)
    conversation, _ = store_module.read(service.chats(), key)
    history = list(conversation.messages)
    window = mc_llm_window.for_thread(key.character, key.thread_id)
    counter = mc_llm_tokens.for_server(window.ratio)
    every = ops._every_picture()
    cut, _ = mc_llm_window.keep_latest(history, window, counter, keep_tokens, every)
    first = mc_llm_window.start_of(window, history)
    span = history[first:cut]
    if not any(message.text.strip() for message in span):
        raise Refused("Nothing older than that is being sent now. Keep fewer tokens to "
                      "compact more of the conversation.")
    spent = mc_llm_window.costs(history, counter, every, start=first)
    tokens = sum(spent[first:cut])
    character = ops._character(key.character)
    persona = ops._persona()
    extra = ops._window_extra(window, ops._emotion_note(ops._emotion_of(key), character,
                                                        persona, ops._emotion_mode()),
                              ops._injection_note(key))
    words = max(int(target * WORDS_PER_TOKEN), 40)
    asked = instruction(character, persona, words, bool(window.summary))
    wire = _wire(character, persona, extra, span, asked)
    name = _name(key)
    job = Job(name=name, cancel=sessions.Cancellation(), target=target, messages=len(span),
              tokens=tokens, cut=cut, digest=_digest(history, cut))
    with _lock:
        if name in _jobs:
            raise Refused("This conversation is already being compacted.")
        _jobs[name] = job
        _outcomes.pop(name, None)
    _publish(key, job)
    worker = threading.Thread(target=_run, args=(key, job, wire, counter.prompt(wire)[0]),
                              name="model-chain-compact", daemon=True)
    job.worker = worker
    worker.start()
    return {"messages": len(span), "tokens": tokens, "target": target, "cut": cut}


def _publish(key, job: Job | None = None, ended: tuple | None = None) -> None:
    try:
        import mc_llm_conversation_service as service

        payload = {"compacting": job.describe() if job is not None else None}
        if ended is not None:
            payload.update(outcome=ended[0], message=ended[1])
        service.publish(service.CONTEXT, key, **payload)
    except Exception:
        logger.debug("Model Chain: could not tell the pages about a compaction",
                     exc_info=True)


def _acquire(gpu, job: Job) -> bool:
    """The language model's turn, as a reply waits for it; what it waits for is
    the job's status meanwhile."""
    import mc_llm_sessions as sessions

    steps = gpu.acquire()
    while True:
        try:
            event = next(steps)
        except StopIteration as finished:
            return bool(finished.value)
        if getattr(event, "kind", "") == sessions.STATUS and getattr(event, "text", ""):
            job.status = event.text


def _run(key, job: Job, wire: list, prompt_tokens: int) -> None:
    from prompt_master.core.models import draw_seed

    import mc_llm_conversation_ops as ops
    import mc_llm_sessions as sessions

    gpu = sessions._Gpu(LABEL, job.cancel)
    ended = ("failed", "The conversation could not be compacted.")
    began = time.monotonic()
    try:
        if not _acquire(gpu, job) or job.cancel.is_set():
            ended = ("cancelled", "Compacting was cancelled. Nothing was changed.")
            return
        job.phase, job.status = "preparing", "Preparing…"
        _publish(key, job)
        client = sessions._client(False, cancel=job.cancel.event)
        job.phase, job.status = "writing", "Summarizing…"
        _publish(key, job)
        last = [time.monotonic()]

        def heard(piece):
            job.written += len(piece or "")
            now = time.monotonic()
            if now - last[0] >= PROGRESS_EVERY:
                last[0] = now
                _publish(key, job)

        # Never past the context the server was placed at: the summary is written
        # after the prompt in the same slot.
        room = max(ops._context_size() - prompt_tokens - 16, 64)
        text = client.stream_chat(wire, min(int(job.target * SLACK), room), draw_seed(), heard,
                                  job.cancel.event, temperature=TEMPERATURE, top_p=TOP_P)
        gpu.release()
        if job.cancel.is_set():
            ended = ("cancelled", "Compacting was cancelled. Nothing was changed.")
            return
        summary = clean(text)
        if not summary:
            ended = ("failed", "The model wrote no summary. Nothing was changed.")
            return
        ended = _keep(key, job, summary)
        if ended[0] == "done":
            logger.info("Model Chain: compacted a conversation — %d messages (%s tokens) into "
                        "a summary of %s tokens in %.1fs", job.messages, f"{job.tokens:,}",
                        f"{_count(summary):,}", time.monotonic() - began)
    except Exception as exc:
        logger.warning("Model Chain: compacting a conversation failed (%s)",
                       type(exc).__name__)
        logger.debug("Model Chain: compacting a conversation failed", exc_info=True)
    finally:
        gpu.release()
        with _lock:
            if _jobs.get(job.name) is job:
                del _jobs[job.name]
            _outcomes[job.name] = (ended[0], ended[1], time.monotonic())
        _publish(key, None, ended)
        if ended[0] == "done":
            _after(key)


def _count(summary: str) -> int:
    import mc_llm_tokens

    return mc_llm_tokens.for_server().text(summary)


def _keep(key, job: Job, summary: str) -> tuple:
    """Keep the summary if the messages it was written from are still there."""
    import mc_llm_conversation_service as service
    import mc_llm_conversation_store as store_module
    import mc_llm_window

    conversation, _ = store_module.read(service.chats(), key)
    history = list(conversation.messages)
    if len(history) < job.cut or _digest(history, job.cut) != job.digest:
        return ("failed", "The conversation changed while it was being compacted. Nothing "
                          "was changed.")
    tokens = _count(summary)
    mc_llm_window.set_summary(key.character, key.thread_id, summary, job.cut, tokens)
    return ("done", f"Compacted {job.messages} messages ({job.tokens:,} tokens) into a "
                    f"summary of {tokens:,}.")


def _after(key) -> None:
    """The thread is read again under its new front, and the pages are told."""
    try:
        import mc_llm_conversation_ops as ops
        import mc_llm_conversation_service as service

        ops.rewarm(key)
        service.publish(service.CONVERSATION_CHANGED, key, kind_detail="window")
    except Exception:
        logger.debug("Model Chain: could not read the compacted conversation again",
                     exc_info=True)


__all__ = ["LABEL", "Refused", "SUMMARY_DEFAULT", "SUMMARY_MAX", "SUMMARY_MIN", "cancel",
           "clean", "instruction", "progress", "reset", "running", "start"]
