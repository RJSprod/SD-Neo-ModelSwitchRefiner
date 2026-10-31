"""What the conversation store does once, when the process comes up.

Four small jobs, none of which may fail the host:

*Report orphans.* Tier 1 durability (specification 5.7) accepts that a crash
between reserving a chat's name and writing it leaves an empty file. It is
never a corrupt conversation -- there was nothing in it -- and it is reported
rather than deleted, because this code cannot tell an interrupted create from a
chat somebody is halfway through importing.

*Mark interrupted replies.* A reply that was in flight when the process went
down has a checkpoint and no terminal phase. It is offered, with its last
checkpoint and the input it was answering, and it is never regenerated
automatically: a machine that resends prompts after a restart is a machine that
bills somebody for work they did not ask for twice.

*Sweep staged uploads.* Pictures that were attached to a message nobody sent,
older than a day, and not pinned by an operation that is still running.

*Clean leaked emotion lines.* Once, ever: the replies saved while llama-server's
echo of a thread's emotion line was kept as their start lose it
(:func:`mc_llm_emotions.clean_saved_replies`).

All four are best-effort and all four are logged rather than raised. The
extension has to start on a machine whose chats folder is on a network drive
that is not mounted yet.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_done = False


def on_app_started(_demo=None, _app=None) -> bool:
    """``script_callbacks.on_app_started``'s signature. Runs once per process."""
    global _done

    if _done:
        return True
    _done = True
    _orphans()
    _interrupted()
    _staged()
    _emotion_lines()
    return True


def _orphans() -> None:
    try:
        import mc_llm_conversation_service as service
        import mc_llm_conversation_store as store

        store.report_orphans(service.chats())
    except Exception:
        logger.debug("Model Chain: could not look for interrupted conversation writes",
                     exc_info=True)


def _interrupted() -> None:
    try:
        import mc_llm_conversation_ops as ops

        found = ops.interrupted()
    except Exception:
        logger.debug("Model Chain: could not look for interrupted replies", exc_info=True)
        return
    if found:
        logger.info("Model Chain: %d reply/replies were interrupted by a restart. They are "
                    "offered with what had arrived; nothing is regenerated automatically",
                    len(found))


def _staged() -> None:
    try:
        import mc_llm_attachment_staging as staging

        gone = staging.expire()
    except Exception:
        logger.debug("Model Chain: could not sweep staged attachments", exc_info=True)
        return
    if gone:
        logger.info("Model Chain: forgot %d staged picture(s) nobody attached to a message",
                    gone)


def _emotion_lines() -> None:
    try:
        import mc_llm_conversation_service as service
        import mc_llm_emotions

        mc_llm_emotions.clean_saved_replies(service.chats())
    except Exception:
        logger.debug("Model Chain: could not clean leaked emotion lines", exc_info=True)
