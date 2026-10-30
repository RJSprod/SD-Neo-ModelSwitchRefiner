"""The cache update: a reply written with an emotion, read again without it.

A thread's emotion (:mod:`mc_llm_emotions`) starts the character's turn on the
wire and is never in the reply. llama-server's prompt cache is the
conversation *as it was read*, so after such a reply it holds the line, then
the reply; and the next prompt has the reply alone in that place. A prompt is
resumed at the longest common prefix, so the next message would read the
whole last reply again before its first word -- on the Intel GPU, a few
seconds for every message while an emotion is on, which is the cost handoff
30 took out of every turn (:mod:`mc_llm_template`).

So the moment such a reply is saved, the conversation as it now stands --
ending on that reply -- is sent again on a thread nobody waits for, with one
token asked for and thrown away. llama-server reads it from where it parts
from its cache (the start of the reply) to its end and keeps it, and the next
message reads only itself. The reply's own slot is not saved to disk (it holds
the line); the slot after this is (:mod:`mc_llm_slot_cache`), so a restart
resumes the conversation as it would have without an emotion.

**The request ends on the reply**, which llama-server continues rather than
closes, and so it renders the reply exactly as the next prompt will render it
as a past turn -- on a template where that is so. Before anything is read,
llama-server is asked to render both (``/apply-template``, which is how the
request itself will be rendered), and the update is made only when the one is
the start of the other. Gemma 4's own template is not (it drops the empty
thought marker from past replies; :mod:`mc_llm_template` keeps it, and is on
by default), and on such a template every turn reads the last reply again
anyway, emotion or not: nothing is read for nothing.

**Nothing is ever started or waited on for it.** Only the server already up
from the settings in force is used (``warm_client``); it gives way to an image
job on its card and to a speech guest's turn, and takes the language model's
lock as background work, which never queues ahead of anything. Skipped, the
next message simply reads the reply again, as it would have.

**A reply in the same conversation waits for it**, and says so at once:
*Waiting for the cache update…* (:func:`wait`), which is the whole point --
started beside the update in another llama-server slot, it would read the
whole conversation. Anything else that wants the language model meanwhile
waits for its lock under the same name. The pages are told it is running and
when it is done (``cache_update`` on the conversation feed, and on the
snapshot), and draw a quiet *Updating cache…* while it is.
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
"""Seconds a reply waits for its conversation's update before going ahead
anyway. An update is a few seconds of reading; this is for one that hung."""

LOCK_WAIT = 2.0
"""Seconds the update waits for the language model's lock before giving up."""

POLL = 0.1

RENDER_TIMEOUT = 15.0

PROBE = "."
"""A message of yours after the reply, for the full rendering the continued
one is compared with. What it says does not matter; that it follows does."""

_guard = threading.Lock()
_pending: dict[str, threading.Event] = {}


def pending(conversation: str) -> bool:
    """Whether this conversation's last reply is being read again right now."""
    with _guard:
        done = _pending.get(str(conversation or ""))
    return done is not None and not done.is_set()


def wait(conversation: str, cancel):
    """Generator: wait for this conversation's update, saying so at once.

    Yields one status event when there is something to wait for, and returns
    True when the reply may go ahead -- the update finished, or took longer
    than :data:`WAIT_LIMIT` -- or False when the reply was stopped meanwhile.
    """
    with _guard:
        done = _pending.get(str(conversation or ""))
    if done is None or done.is_set():
        return True
    import mc_llm_sessions as sessions

    yield sessions.Event(sessions.STATUS, WAITING)
    began = time.monotonic()
    while not done.wait(POLL):
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
    moment later finds it and waits. False when one is already running.
    """
    import mc_llm_conversation_service as service

    name = service.conversation_name(key)
    done = threading.Event()
    with _guard:
        running = _pending.get(name)
        if running is not None and not running.is_set():
            return False
        _pending[name] = done
    service.publish(service.CACHE_UPDATE, key, updating=True)
    worker = threading.Thread(target=_run, args=(key, name, build, done),
                              name="model-chain-cache-update", daemon=True)
    worker.start()
    return True


def _run(key, name: str, build, done: threading.Event) -> None:
    import mc_llm_conversation_service as service

    try:
        request = build()
        said = _update(request)
        if said:
            logger.info("Model Chain: %s", said)
    except Exception:
        logger.debug("Model Chain: the cache update failed", exc_info=True)
    finally:
        with _guard:
            if _pending.get(name) is done:
                del _pending[name]
        done.set()
        service.publish(service.CACHE_UPDATE, key, updating=False)


def _update(request) -> str:
    """Read ``request`` into llama-server's cache, or say why not. Never raises
    for a reason it expects; the console line is the answer."""
    import mc_broker
    import mc_llm_runtime
    import mc_llm_sessions as sessions

    chosen = sessions._runtime_for("")
    warm = getattr(chosen.runtime, "warm_client", None)
    client = warm(request.needs_vision) if callable(warm) else None
    if client is None:
        return ""
    if mc_llm_runtime._stands_down_for_the_image_job(chosen.configuration):
        return ("the cache update after a reply with an emotion gave way to the image "
                "generation; the next message reads the last reply again")
    domain = mc_llm_runtime.execution_domain(chosen.configuration)
    if sessions._Gpu._blocked_by_a_guest(domain, False):
        return ("the cache update after a reply with an emotion gave way to a Voice Box "
                "render; the next message reads the last reply again")
    with mc_broker.workload(mc_broker.FAMILY_LLM, LABEL, timeout=LOCK_WAIT, background=True,
                            required=False, domain=domain) as held:
        if not held:
            return ("the cache update after a reply with an emotion found the language model "
                    "busy; the next message reads the last reply again")
        if not continues(client, request.messages):
            logger.debug("Model Chain: this chat template does not render a continued reply "
                         "as a past one, so there is no cache update to make")
            return ""
        began = time.monotonic()
        saving = getattr(client, "stream_conversation", None)
        if request.conversation and callable(saving):
            saving(request.conversation, request.messages, 1, request.seed, _discard, None,
                   temperature=request.temperature, top_p=request.top_p)
        else:
            client.stream_chat(request.messages, 1, request.seed, _discard,
                               temperature=request.temperature, top_p=request.top_p)
        return (f"the cache update read the last reply again in "
                f"{time.monotonic() - began:.1f}s; the next message reads only what is new")


def _discard(_text) -> None:
    pass


def continues(client, messages) -> bool:
    """Whether the request ending on a reply renders as the start of the request
    that follows it -- the reply as a past turn, then a message of yours.

    Asked of llama-server itself, with the request's own template fields, so the
    answer is about the rendering the requests will get.
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
        for done in _pending.values():
            done.set()
        _pending.clear()


__all__ = ["LABEL", "WAITING", "WAIT_LIMIT", "continues", "pending", "reset", "schedule",
           "wait"]
