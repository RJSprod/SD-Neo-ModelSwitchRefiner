"""The cache update: a reply written with an emotion, read again without it.

A thread's emotion (:mod:`mc_llm_emotions`) starts the character's turn on the
wire and is never in the reply. llama-server's prompt cache is the
conversation *as it was read*, so after such a reply it holds the line, then
the reply; and the next prompt has the reply alone in that place. A prompt is
resumed at the longest common prefix, so the next message would read the
whole last reply again before its first word -- on the Intel GPU, a few
seconds for every message while an emotion is on, which is the cost handoff
30 took out of every turn (:mod:`mc_llm_template`).

So the moment such a reply is saved, the conversation as it now stands is sent
again on a thread nobody waits for, with one token asked for and thrown away.
llama-server reads it from where it parts from its cache (the start of the
reply) and keeps it, and the next message reads only itself.

**It follows the language model's own rules, whatever the model.** It waits
for its turn exactly as a reply does -- the same acquisition
(``mc_llm_sessions._Gpu``): for an image generation on its card, a Voice Box
render, another language-model job -- under the name *the cache update*, and
it gives up only when a message in the same conversation takes its place
(below) or when the server that wrote the reply is no longer running, in
which case there is no cache left to update and the next message reads the
conversation as it would without an emotion. It never starts or restarts a
server (``warm_client``).

**Two forms, the same reading.** The request is the conversation the next
message will extend, and llama-server must write the reply into its cache
exactly as the next prompt will have it. Where the chat template renders a
reply that a request ends on (which llama-server continues) the same way as a
past reply -- asked of llama-server itself, ``/apply-template`` with the
request's own template fields (:func:`continues`) -- the request ends on the
reply, and its slot is saved, so a restart resumes exactly (Gemma 4 with its
thought marker kept, :mod:`mc_llm_template`, is such a template). Everywhere
else -- Gemma 4's own template, which drops the empty thought marker from past
replies, or any other that writes the newest reply differently -- the reply is
sent finished, followed by a placeholder message of yours: every template
renders a past reply the same way in both, and llama-server sets a resume
point at the start of the last message of yours for a model that cannot roll
its cache back freely (a sliding window without the full cache, a hybrid), so
the next message resumes exactly where its own words begin. That slot is not
saved: it holds the placeholder.

**A message in the same conversation** waits for an update that is reading,
and says so at once: *Waiting for the cache update…* (:func:`wait`). One sent
while the update is still waiting for its turn takes its place instead -- it
reads the same reply as part of its own prompt, so nothing is waited for
twice. The pages are told an update is pending and when it ends
(``cache_update`` on the conversation feed and the snapshot), and draw a quiet
*Updating cache…* meanwhile.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("model_chain")

LABEL = "the cache update"
"""The language model's lock is held under this name, so anything that waits
for it is told *Waiting for the cache update…*."""

WAITING = "Waiting for the cache update…"

WAIT_LIMIT = 180.0
"""Seconds a reply waits for its conversation's update to finish reading before
going ahead anyway. A reading is a few seconds; this is for one that hung."""

POLL = 0.1

RENDER_TIMEOUT = 15.0

PROBE = "."
"""The placeholder message of yours after the reply, in the form that sends the
reply finished, and in the rendering the continued one is compared with. What
it says does not matter; that it follows does."""


class _Update:
    """One conversation's update: whether it has ended, whether it has begun
    reading, and the stop a message that takes its place presses."""

    def __init__(self):
        import mc_llm_sessions as sessions

        self.done = threading.Event()
        self.reading = False
        self.cancel = sessions.Cancellation()


_guard = threading.Lock()
_pending: dict[str, _Update] = {}


def pending(conversation: str) -> bool:
    """Whether this conversation's last reply is waiting to be, or being, read
    again right now."""
    with _guard:
        update = _pending.get(str(conversation or ""))
    return update is not None and not update.done.is_set()


def wait(conversation: str, cancel):
    """Generator: let a reply in this conversation go ahead, or wait for it.

    An update still waiting for its turn is stopped -- the reply reads the
    same reply in its own prompt -- and the reply goes ahead at once, saying
    nothing. An update that is reading is waited for, and the reply says so in
    one status event first. Returns True when the reply may go ahead, False
    when it was stopped meanwhile; after :data:`WAIT_LIMIT` it goes ahead.
    """
    with _guard:
        update = _pending.get(str(conversation or ""))
        if update is None or update.done.is_set():
            return True
        if not update.reading:
            update.cancel.cancel()
            return True
    import mc_llm_sessions as sessions

    yield sessions.Event(sessions.STATUS, WAITING)
    began = time.monotonic()
    while not update.done.wait(POLL):
        if cancel is not None and cancel.is_set():
            return False
        if time.monotonic() - began > WAIT_LIMIT:
            logger.info("Model Chain: a reply waited %.0f s for the cache update and went "
                        "ahead without it", WAIT_LIMIT)
            return True
    return True


def schedule(key, build) -> bool:
    """Read this conversation's last reply again, on a thread of its own.

    ``build`` makes the request -- the conversation ending on the reply -- and
    runs on that thread. Registered before this returns, so a reply asked for a
    moment later finds it. False when one is already pending.
    """
    import mc_llm_conversation_service as service

    name = service.conversation_name(key)
    update = _Update()
    with _guard:
        running = _pending.get(name)
        if running is not None and not running.done.is_set():
            return False
        _pending[name] = update
    service.publish(service.CACHE_UPDATE, key, updating=True)
    worker = threading.Thread(target=_run, args=(key, name, build, update),
                              name="model-chain-cache-update", daemon=True)
    worker.start()
    return True


def _run(key, name: str, build, update: _Update) -> None:
    import mc_llm_conversation_service as service

    try:
        request = build()
        said = _update(request, update)
        if said:
            logger.info("Model Chain: %s", said)
    except Exception:
        logger.debug("Model Chain: the cache update failed", exc_info=True)
    finally:
        with _guard:
            if _pending.get(name) is update:
                del _pending[name]
        update.done.set()
        service.publish(service.CACHE_UPDATE, key, updating=False)


def _warm(request):
    """A client for the server that wrote the reply, or None: never a start."""
    import mc_llm_sessions as sessions

    runtime = sessions._runtime_for("").runtime
    warm = getattr(runtime, "warm_client", None)
    return warm(request.needs_vision) if callable(warm) else None


def _drive(steps) -> bool:
    """Run a generator that yields status events to its end; its answer."""
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return bool(finished.value)


def _update(request, update: _Update) -> str:
    """Read ``request`` into llama-server's cache when the language model's turn
    comes, or say why not. The console line is the answer; nothing raises for a
    reason it expects."""
    import mc_llm_sessions as sessions

    if _warm(request) is None:
        return ""
    gpu = sessions._Gpu(LABEL, update.cancel)
    try:
        # A reply's own wait: the image job on this card, a speech guest's
        # turn, the lock. Stopped only by a message that takes its place.
        if not _drive(gpu.acquire()):
            return ""
        # Asked again with the card held: the server may have been stopped or
        # replaced while this waited, and then there is no cache to update.
        client = _warm(request)
        if client is None:
            return ""
        with _guard:
            if update.cancel.is_set():
                return ""
            update.reading = True
        began = time.monotonic()
        continued = continues(client, request.messages)
        messages = list(request.messages)
        kept = {}
        if not continued:
            messages.append({"role": "user", "content": PROBE})
            kept = {"save": False}
        saving = getattr(client, "stream_conversation", None)
        if request.conversation and callable(saving):
            saving(request.conversation, messages, 1, request.seed, _discard, None,
                   temperature=request.temperature, top_p=request.top_p, **kept)
        else:
            client.stream_chat(messages, 1, request.seed, _discard,
                               temperature=request.temperature, top_p=request.top_p)
        return (f"the cache update read the last reply again in "
                f"{time.monotonic() - began:.1f}s"
                + ("" if continued else ", as a past reply")
                + "; the next message reads only what is new")
    finally:
        gpu.release()


def _discard(_text) -> None:
    pass


def continues(client, messages) -> bool:
    """Whether the request ending on a reply renders as the start of the request
    that follows it -- the reply as a past turn, then a message of yours.

    Asked of llama-server itself, with the request's own template fields, so the
    answer is about the rendering the requests will get. No answer is a no: the
    reply is then sent finished, which every template renders alike.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return False
    continued = _render(client, messages)
    followed = _render(client, list(messages) + [{"role": "user", "content": PROBE}])
    return bool(continued) and bool(followed) and followed.startswith(continued) \
        and len(followed) > len(continued)


def _render(client, messages) -> str:
    import httpx

    body = {"messages": messages, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    try:
        with httpx.Client(timeout=httpx.Timeout(RENDER_TIMEOUT, connect=5),
                          headers={"Authorization": f"Bearer {client.api_key}"}) as http:
            response = http.post(f"{client.base_url}/apply-template", json=body)
            response.raise_for_status()
            found = response.json()
    except Exception:
        logger.debug("Model Chain: llama-server could not render a request for the cache "
                     "update", exc_info=True)
        return ""
    prompt = found.get("prompt") if isinstance(found, dict) else None
    return prompt if isinstance(prompt, str) else ""


def reset() -> None:
    """Forget every update in flight. For the tests."""
    with _guard:
        for update in _pending.values():
            update.cancel.cancel()
            update.done.set()
        _pending.clear()


__all__ = ["LABEL", "PROBE", "WAITING", "WAIT_LIMIT", "continues", "pending", "reset",
           "schedule", "wait"]
