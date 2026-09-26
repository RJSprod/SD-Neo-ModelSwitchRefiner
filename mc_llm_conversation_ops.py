"""A reply is a thing the server owns, not a thing a browser is holding open.

At HEAD the reply existed only inside a Gradio generator. Stop was wired as
``cancels=``, which *closes* that generator — ``GeneratorExit``, a
``BaseException``, straight past every ``except Exception`` in it — so the only
reason a stopped reply survived at all was a ``finally`` block written after a
bug report that said replies were disappearing. A browser refresh went through
the same hole. So did a dropped queue entry. So did a second window.

Here the reply runs on a service thread that owns its own cancellation and its
own checkpoints, and both views *subscribe*. The Gradio generator becomes a
follower that writes nothing: closing it — by refreshing, by ``cancels=``, by
navigating away — loses a subscription and nothing else. This is not a new
pattern in this repository; :mod:`mc_llm_jobs` has run MiniMax's requests this
way since it was written.

Three guarantees, and where each lives
--------------------------------------
*The reply is saved exactly once, into the conversation it was asked for.* The
completion transaction compares against the revision the operation was accepted
at. If the conversation moved — an edit from another window, a delete, a
branch — the reply is **not** written. It goes to the recovery store and the
person is offered it. A reply saved into a conversation that is no longer the
one it answers is worse than a reply that needs a click.

*Stop means stopped.* One cancellation token per operation, set by whoever asks,
honoured by ``mc_llm_sessions`` between tokens exactly as it always was. Once
Stop is accepted for a running operation, no automatic speech may begin — that
race had two winners before and the audio was one of them.

*Two replies cannot consume each other's completion.* The process-global
one-shot slot that ``take_completed_reply()`` popped is gone. A completion
record belongs to an operation id and is *read*, never consumed, so two threads
finishing at once cannot take each other's answer and a duplicate terminal
callback -- which a host is entitled to deliver -- gets the same answer twice
rather than somebody else's once.

What did not move
-----------------
The prompt builder, the sampler, the trimmer, the model loader, the resource
arbitration and every speech engine. This module calls
``mc_llm_sessions.conversation(request, cancel)`` with a request built by
``prompt_master.chat.prompt.build`` — the same two calls the panel made, in the
same order, with the same arguments. It is not a scheduler and does not decide
who gets the card; ``mc_llm_sessions._Gpu`` still does.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mc_llm_conversation_store as store_module
from mc_llm_conversation_store import Key, Receipt

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

# -- phases ----------------------------------------------------------------- #

ACCEPTED = "accepted"
WAITING = "waiting"
PREPARING = "preparing"
GENERATING = "generating"
SAVING = "saving"
COMPLETED = "completed"
STOPPING = "stopping"
STOPPED = "stopped"
FAILED = "failed"
SAVE_FAILED = "save_failed"
INTERRUPTED = "interrupted"

TERMINAL = (COMPLETED, STOPPED, FAILED, SAVE_FAILED, INTERRUPTED)

PHASE_TEXT = {
    ACCEPTED: "Starting…",
    WAITING: "Waiting for resources…",
    PREPARING: "Preparing model…",
    GENERATING: "Generating…",
    SAVING: "Saving…",
    COMPLETED: "Reply complete.",
    STOPPING: "Stopping…",
    STOPPED: "Stopped.",
    FAILED: "That did not work.",
    SAVE_FAILED: "Response not saved.",
    INTERRUPTED: "Reply interrupted by server restart.",
}
"""One line per phase, said the same way in both views.

Never "Ready" while an operation is waiting on a resource, which is what the
tab used to show between the queue accepting a run and llama-server answering
-- and that gap is measured in tens of seconds on a cold model.
"""

# -- how a target is written back ------------------------------------------- #

APPEND = "append"
"""The reply is a new message at the end. ``send``, and both branching paths."""

VERSION = "version"
"""The reply is a new version of a message that is already there. Final regenerate."""

IN_PLACE = "in_place"
"""The reply extends the selected version of a message. ``continue``."""

CHECKPOINT_INTERVAL = 1.0
"""How often a running reply's text is written down. At most once a second.

A checkpoint is not a conversation write -- it never touches the chat file and
never moves the revision. It is what a restart can show you of a reply that was
in flight, and the bound on what it costs to lose is therefore one second of
tokens.
"""

MAX_RECOVERY = 20
MAX_RECOVERY_BYTES = 32 * 1024 * 1024
"""How much unsaved generated text is kept for explicit recovery.

Reserved *before* work is accepted rather than checked afterwards: a store that
discovers it is full at the moment a reply needs preserving is a store that
loses the reply it exists to keep.
"""

RECOVERY_TTL = 86400.0
"""How long an unresolved result is kept without being asked about. One day.

Restart loses them anyway (7.6), and the copy says so rather than implying a
durability this tier does not have.
"""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class Completion:
    """What one finished operation produced. Read by both views; consumed by none.

    Replaces the process-global one-shot slot (``_completed`` /
    ``take_completed_reply``) that two pages could pop from each other. Bound to
    an operation id, so concurrent replies in two threads cannot take each
    other's -- which is the defect, not a hypothetical.
    """

    operation_id: str
    character: str = ""
    thread_id: str = ""
    final_text: str = ""
    speech_turn_id: str = ""
    status: str = COMPLETED
    auto_speech_owner_page: str = ""
    at: float = field(default_factory=time.time)

    def describe(self) -> dict:
        return {"operation_id": self.operation_id, "status": self.status,
                "conversation": {"character": self.character, "thread_id": self.thread_id},
                "speech_turn_id": self.speech_turn_id,
                "auto_speech_owner_page": self.auto_speech_owner_page,
                "final_text": self.final_text, "at": self.at}


@dataclass
class Unresolved:
    """A finished reply that could not be written where it belonged.

    Held so that the person decides what happens to it. Never auto-branched,
    never auto-resent, never spoken: a result that lost its race is a result
    somebody has to look at, and doing anything with it silently is how the
    conversation they were actually having gets a paragraph they did not ask
    for.
    """

    operation_id: str
    character: str
    thread_id: str
    accepted_revision: object
    text: str
    target_index: int
    target_kind: str
    reason: str = "result_conflict"
    at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.text.encode("utf-8"))

    def describe(self) -> dict:
        return {"operation_id": self.operation_id,
                "conversation": {"character": self.character, "thread_id": self.thread_id},
                "accepted_revision": self.accepted_revision, "text": self.text,
                "target_index": self.target_index, "reason": self.reason, "at": self.at}


@dataclass
class Operation:
    """One reply being produced, from acceptance to terminal.

    Everything a subscriber needs is on this object, and nothing a subscriber
    needs is anywhere else -- which is what makes a follower able to write
    nothing at all.
    """

    operation_id: str
    key: Key
    action: str
    target_index: int
    target_kind: str = APPEND
    target_version: int = 0
    accepted_revision: object = None
    opening: str = ""
    page_id: str = ""
    voice_wanted: bool = False
    phase: str = ACCEPTED
    status_text: str = PHASE_TEXT[ACCEPTED]
    text: str = ""
    error: str = ""
    seq: int = 0
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    accepted_fingerprint: str = ""
    """The exact bytes of the conversation when this operation was accepted.

    Carried as well as the revision because the two catch different things. The
    revision catches another window's write, which is the common case and the
    one the whole design is for. The fingerprint catches a file that changed
    without going through this code at all -- an editor, a sync client, a second
    process -- which leaves the counter exactly where it was and the
    conversation something else entirely.
    """

    request: object = None
    cancel: object = None
    turn: object = None
    speech_turn_id: str = ""
    stop_requested: bool = False
    result_revision: int = 0
    recovery: str = ""

    def describe(self) -> dict:
        return {"operation_id": self.operation_id, "phase": self.phase,
                "status": self.status_text,
                "conversation": {"character": self.key.character,
                                 "thread_id": self.key.thread_id},
                "target_index": self.target_index, "target_kind": self.target_kind,
                "target_version": self.target_version,
                "generated_text": self.text, "provisional": self.phase not in TERMINAL,
                "operation_seq": self.seq, "accepted_revision": self.accepted_revision,
                "speech_turn_id": self.speech_turn_id,
                "recovery_status": self.recovery or None,
                "result_revision": self.result_revision or None,
                "error": self.error or None,
                "stop_requested": self.stop_requested,
                # How long this has been going, from acceptance: what a panel
                # can put beside "Replying…" so that three minutes of prompt
                # reading looks like three minutes and not like a hang.
                "elapsed": round(max(time.time() - self.started, 0.0), 1)}


# --------------------------------------------------------------------------- #
# The register
# --------------------------------------------------------------------------- #

_guard = threading.Condition()
_operations: dict[str, Operation] = {}
_completions: dict[str, Completion] = {}
_unresolved: dict[str, Unresolved] = {}


def get(operation_id: str) -> Operation | None:
    with _guard:
        return _operations.get(str(operation_id or ""))


def completion(operation_id: str) -> Completion | None:
    """One finished operation's result. Read, never consumed."""
    with _guard:
        return _completions.get(str(operation_id or ""))


def for_conversation(key: Key) -> Operation | None:
    with _guard:
        for operation in _operations.values():
            if operation.key == key and operation.phase not in TERMINAL:
                return operation
        return None


def running() -> list:
    with _guard:
        return [operation.describe() for operation in _operations.values()
                if operation.phase not in TERMINAL]


def recoverable(key: Key | None = None) -> list:
    with _guard:
        return [item for item in _unresolved.values()
                if key is None or (item.character == key.character
                                   and item.thread_id == key.thread_id)]


def drain(timeout: float = 30.0) -> bool:
    """Wait for every running reply to reach a terminal phase. ``True`` if all did.

    Used by shutdown, so a reload does not abandon a reply that is one token
    from being saved, and by tests, which have to be able to ask "and then what
    happened" about work that is deliberately no longer on the caller's thread.

    It waits; it does not cancel. A caller that wants the replies stopped calls
    :func:`stop` first and then this.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    with _guard:
        while any(operation.phase not in TERMINAL for operation in _operations.values()):
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            _guard.wait(timeout=min(left, 0.2))
    return True


def reset() -> None:
    """Drop everything this process knows about replies. Tests, and UI reload."""
    with _guard:
        _operations.clear()
        _completions.clear()
        _unresolved.clear()
        _guard.notify_all()


def _publish(operation: Operation, kind: str, force: bool = False, **payload) -> None:
    import mc_llm_conversation_service as service

    service.publish(kind, operation.key, operation_id=operation.operation_id,
                    operation_seq=operation.seq, force=force, **payload)


def _advance(operation: Operation, phase: str, status: str = "", error: str = "") -> None:
    """Move one operation to a phase and tell everybody. Under the lock.

    The sequence number moves with the phase, which is what lets a subscriber
    discard an event that arrived out of order without having to reason about
    what the phase means.
    """
    import mc_llm_conversation_service as service

    with _guard:
        if operation.phase in TERMINAL and phase not in TERMINAL:
            return
        operation.phase = phase
        operation.status_text = status or PHASE_TEXT.get(phase, phase)
        if error:
            operation.error = error
        operation.seq += 1
        if phase in TERMINAL:
            operation.finished = time.time()
        _guard.notify_all()
    _publish(operation,
             service.OPERATION_TERMINAL if phase in TERMINAL else service.OPERATION_PHASE,
             force=True, phase=operation.phase, status=operation.status_text,
             text=operation.text, error=operation.error or None,
             revision=operation.result_revision or None)


# --------------------------------------------------------------------------- #
# Subscribing
# --------------------------------------------------------------------------- #


def listen(operation_id: str, idle: float = 0.2, timeout: float | None = None):
    """Yield this operation's state every time it changes, until it ends.

    What the tab's Gradio generator follows. It writes nothing, saves nothing
    and cancels nothing; its ``finally`` closes a subscription and that is all,
    so a refresh mid-reply costs the view and not the reply.

    The first yield is the current state rather than the next change: a
    follower attached after the operation started must not wait for a token to
    find out what is happening.
    """
    started = time.monotonic()
    seen = -1
    while True:
        with _guard:
            operation = _operations.get(str(operation_id or ""))
            if operation is None:
                return
            if operation.seq == seen and operation.phase not in TERMINAL:
                if timeout is not None and time.monotonic() - started >= timeout:
                    return
                _guard.wait(timeout=idle)
                operation = _operations.get(str(operation_id or ""))
                if operation is None:
                    return
            found = operation.describe()
            seen = operation.seq
            done = operation.phase in TERMINAL
        yield found
        if done:
            return


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #


def begin(envelope, scope: str) -> dict:
    """Accept one generating command, then start it. Returns before inference.

    The order is the promise. Everything that can refuse -- a stale revision, a
    busy thread, an attachment that is not ready, a model with no projector --
    refuses here, with the conversation locked, before a single token is asked
    for. Once this returns ``accepted`` the reply is the server's problem and no
    browser can lose it.
    """
    import mc_llm_conversation_service as service

    store = service.chats()
    key = envelope.conversation
    action = envelope.action

    if for_conversation(key) is not None:
        owner = for_conversation(key)
        raise service.Refused(service.THREAD_BUSY,
                              "Reply in progress. Stop it before changing this conversation.",
                              operation_id=owner.operation_id if owner else "")
    if not _reserve_recovery():
        raise service.Refused(service.SERVICE_UNAVAILABLE,
                              "There are unsaved replies waiting for a decision. "
                              "Save or discard one before starting another.")

    plan = _plan(store, envelope)
    operation = Operation(operation_id=envelope.operation_id, key=plan["key"],
                          action=action, target_index=plan["target_index"],
                          target_kind=plan["target_kind"],
                          target_version=plan["target_version"],
                          accepted_revision=plan["accepted_revision"],
                          accepted_fingerprint=_fingerprint(store, plan["key"]),
                          opening=plan["opening"], page_id=envelope.page_id,
                          voice_wanted=bool(envelope.payload.get("voice", True)),
                          request=plan["request"])

    import mc_llm_sessions as sessions

    operation.cancel = sessions.Cancellation()
    # The speech turn is created here, before the executor starts, because the
    # browser has to have its id in the very first frame it is sent: that is
    # what lets it open the audio stream while the model is still thinking. A
    # turn created on the executor thread would be a turn the first few frames
    # do not know about, and a frame carrying an empty id reads as Voice
    # silently switched off.
    #
    # Failure-tolerant throughout: a turn that cannot be created is a reply
    # that streams exactly as it always did. Voice never takes Conversation
    # down with it.
    if operation.voice_wanted:
        import mc_voice_ui

        try:
            operation.turn = mc_voice_ui.begin_speech(character=plan["character"],
                                                      persona=plan["persona"],
                                                      opening=plan["opening"])
        except Exception:
            logger.debug("Model Chain: could not begin speech for this reply", exc_info=True)
    operation.speech_turn_id = getattr(operation.turn, "id", "") or ""
    store_module.claim(store, operation.key, operation.operation_id,
                       {"action": action, "page_id": envelope.page_id})
    with _guard:
        _operations[operation.operation_id] = operation
    _publish(operation, service.OPERATION_ACCEPTED, force=True, phase=ACCEPTED,
             status=operation.status_text, revision=plan["result_revision"] or None,
             target_index=operation.target_index, target_kind=operation.target_kind)

    thread = threading.Thread(target=_execute, args=(operation, plan),
                              name=f"mc-conversation-{operation.operation_id[:8]}",
                              daemon=True)
    thread.start()

    return {"ok": True, "operation_id": operation.operation_id,
            "persisted": bool(plan["persisted"]), "duplicate": False, "phase": ACCEPTED,
            "server_epoch": service.SERVER_EPOCH,
            "accepted_revision": plan["accepted_revision"],
            "resulting_conversation": {"character": operation.key.character,
                                       "thread_id": operation.key.thread_id},
            "revision": plan["result_revision"],
            "target": {"index": operation.target_index, "kind": operation.target_kind,
                       "version": operation.target_version}}


def _fingerprint(store, key: Key) -> str:
    """The conversation's exact bytes, right now. Empty if it cannot be read.

    One extra read, once per operation, off every hot path: it happens between
    accepting the command and starting the model, where the next thing that
    will happen is a model load measured in seconds.
    """
    from prompt_master.chat.history import fingerprint

    try:
        _, raw = store_module.read(store, key)
        return fingerprint(raw)
    except Exception:
        logger.debug("Model Chain: could not fingerprint a conversation", exc_info=True)
        return ""


def _plan(store, envelope) -> dict:
    """The acceptance transaction, and the immutable snapshot of what to ask.

    "Immutable" is the load-bearing word (7.3). Everything the request is built
    from -- the messages up to the target, the character, the persona, the
    sampling, the prefix -- is captured here, under the lock, and the model is
    asked from that capture. A conversation edited while a reply streams
    therefore produces the reply that was asked for, not one built from a
    history that changed underneath it.
    """
    import mc_llm_conversation_service as service

    action = envelope.action
    if action == service.SEND:
        return _plan_send(store, envelope)
    if action == service.REGENERATE:
        return _plan_regenerate(store, envelope)
    if action == service.CONTINUE:
        return _plan_continue(store, envelope)
    if action == service.RESEND_FROM_USER:
        return _plan_resend(store, envelope)
    raise service.Refused(service.INVALID_INPUT, f"{action} does not produce a reply.")


def _plan_send(store, envelope) -> dict:
    from prompt_master.chat.history import USER

    import mc_llm_conversation_service as service

    text = (envelope.payload.get("text") or "").strip()
    token = envelope.payload.get("attachment_token") or ""
    staged = None
    if token:
        import mc_llm_attachment_staging as staging

        staged = staging.resolve(token)
        if staged is None:
            raise service.Refused(service.ATTACHMENT_EXPIRED, "Attach this image again.")
        if not staged.ready:
            raise service.Refused(service.ATTACHMENT_NOT_READY,
                                  "That picture is still being prepared.")
    if not text and staged is None:
        raise service.Refused(service.INVALID_INPUT, "Write a message first.")
    if staged is not None and not _sees():
        raise service.Refused(service.VISION_UNAVAILABLE,
                              "The model running has no vision projector, so the attached "
                              "image cannot be sent to it. Choose one in Setup, or remove "
                              "the image.")

    key = envelope.conversation
    captured: dict = {}
    receipt = Receipt(operation_id=envelope.operation_id, action=envelope.action,
                      issued_at=envelope.issued_at, payload_digest=envelope.digest())

    def change(conversation):
        import mc_llm_attachments

        kept, name = "", ""
        if staged is not None:
            kept = mc_llm_attachments.store(staged.image, conversation.character)
            name = staged.name
        conversation.append(USER, text, image_name=name, image_path=kept)
        captured["target_index"] = len(conversation.messages)
        captured["history"] = list(conversation.messages)
        captured["prefix"] = conversation.response_prefix
        return True

    committed = store_module.transaction(store, key, envelope.expected_revision, change,
                                         receipt=receipt)
    if token:
        import mc_llm_attachment_staging as staging

        staging.consume(token)
    service.publish(service.CONVERSATION_CHANGED, key, revision=committed.revision)
    return _capture(envelope, key, captured["target_index"], APPEND, 0, committed.revision,
                    captured["history"], captured["prefix"], persisted=True,
                    result_revision=committed.revision)


def _plan_regenerate(store, envelope) -> dict:
    """Final reply: a version. Earlier reply: a branch. Existing semantics, guarded."""
    from prompt_master.chat.history import ASSISTANT

    import mc_llm_conversation_service as service

    conversation, raw = store_module.read(store, envelope.conversation)
    current = store_module.token_of(conversation, raw)
    if not store_module.matches(envelope.expected_revision, current):
        raise service.Refused(service.STALE_REVISION,
                              "This conversation changed in another window. Your draft is "
                              "safe.", current_revision=current)
    index = _last_reply(conversation, envelope.target_index)
    if not 0 <= index < len(conversation.messages):
        raise service.Refused(service.INVALID_INPUT, "There is no reply to regenerate.")
    if conversation.messages[index].role != ASSISTANT:
        raise service.Refused(service.INVALID_INPUT,
                              "Regenerate applies to a reply. For one of your own "
                              "messages, use Send again from here.")

    if index < len(conversation.messages) - 1:
        # Branch at the message *before* this reply, so the branch ends on the
        # turn the reply was answering. An opening reply has no turn before it,
        # and branch(-1) is the empty copy that says so.
        made, revision = _branched(store, envelope, index - 1)
        history = list(made.messages)
        return _capture(envelope, _key_of(made), len(made.messages), APPEND, 0, revision,
                        history, made.response_prefix, persisted=True,
                        result_revision=revision)

    message = conversation.messages[index]
    history = list(conversation.messages[:index])
    return _capture(envelope, envelope.conversation, index, VERSION, len(message.versions),
                    current, history, conversation.response_prefix, persisted=False,
                    result_revision=int(conversation.revision or 0))


def _plan_continue(store, envelope) -> dict:
    """Final reply: carry on in place. Earlier reply: carry on in a branch (S8).

    The second half is one of the two authorised behaviour changes. Continuing
    an earlier reply used to mutate a message that had descendants: the replies
    under it went on answering a paragraph that no longer said what they were
    answering, and nothing recorded that it had changed.
    """
    from prompt_master.chat.history import ASSISTANT
    from prompt_master.chat.prompt import continue_instruction

    import mc_llm_conversation_service as service

    conversation, raw = store_module.read(store, envelope.conversation)
    current = store_module.token_of(conversation, raw)
    if not store_module.matches(envelope.expected_revision, current):
        raise service.Refused(service.STALE_REVISION,
                              "This conversation changed in another window. Your draft is "
                              "safe.", current_revision=current)
    index = _last_reply(conversation, envelope.target_index)
    if not 0 <= index < len(conversation.messages):
        raise service.Refused(service.INVALID_INPUT, "There is no reply to continue.")
    message = conversation.messages[index]
    if message.role != ASSISTANT or not message.text.strip():
        raise service.Refused(service.INVALID_INPUT, "There is nothing to carry on from.")

    instruction = continue_instruction(_character(conversation.character))
    if index < len(conversation.messages) - 1:
        made, revision = _branched(store, envelope, index)
        target = len(made.messages) - 1
        carried = made.messages[target]
        return _capture(envelope, _key_of(made), target, IN_PLACE, carried.active, revision,
                        list(made.messages[:target + 1]), made.response_prefix,
                        persisted=True, opening=carried.text, instruction=instruction,
                        result_revision=revision)

    return _capture(envelope, envelope.conversation, index, IN_PLACE, message.active,
                    current, list(conversation.messages[:index + 1]),
                    conversation.response_prefix, persisted=False, opening=message.text,
                    instruction=instruction, result_revision=int(conversation.revision or 0))


def _plan_resend(store, envelope) -> dict:
    """Send again from here — as a branch, never as a truncation (S8).

    The defect this replaces is live data loss at HEAD: ``truncate_after(index)``
    then ``save()`` permanently deleted every message after the selected one.
    It is the same defect ``_regenerate`` was fixed for and the fix was never
    applied here. The original thread is now untouched and the new attempt
    happens in a copy.

    Except for the last message. A message of yours with nothing after it --
    the state a deleted, stopped or failed reply leaves -- has no replies for a
    branch to protect, and copying the whole thread to answer it was a second
    thread nobody asked for, holding the conversation somebody was in the
    middle of. So it is answered in place, the way Send answers: nothing is
    written at acceptance, and the reply is appended under the same revision
    guard every completion has. This is what the flyout's SEND AGAIN is, and
    the tab's "Send again from here" on a last message does the same.
    """
    from prompt_master.chat.history import USER

    import mc_llm_conversation_service as service

    conversation, raw = store_module.read(store, envelope.conversation)
    current = store_module.token_of(conversation, raw)
    if not store_module.matches(envelope.expected_revision, current):
        raise service.Refused(service.STALE_REVISION,
                              "This conversation changed in another window. Your draft is "
                              "safe.", current_revision=current)
    index = envelope.target_index
    if index is None or not 0 <= index < len(conversation.messages):
        raise service.Refused(service.STALE_REVISION, "That message is not there any more.")
    if conversation.messages[index].role != USER:
        raise service.Refused(service.INVALID_INPUT,
                              "Send again from here applies to one of your own messages. "
                              "For a reply, use Regenerate.")

    if index == len(conversation.messages) - 1:
        return _capture(envelope, envelope.conversation, len(conversation.messages),
                        APPEND, 0, current, list(conversation.messages),
                        conversation.response_prefix, persisted=False,
                        result_revision=int(conversation.revision or 0))

    made, revision = _branched(store, envelope, index)
    return _capture(envelope, _key_of(made), len(made.messages), APPEND, 0, revision,
                    list(made.messages), made.response_prefix, persisted=True,
                    result_revision=revision)


def _branched(store, envelope, index: int):
    import mc_llm_conversation_service as service

    committed = service._branch_from(store, envelope.conversation,
                                     envelope.expected_revision, index,
                                     envelope.operation_id, envelope.issued_at,
                                     envelope.digest())
    made = committed.conversation
    service.publish(service.THREAD_CREATED, _key_of(made), revision=committed.revision,
                    branched_from=envelope.conversation.thread_id)
    return made, committed.revision


def _key_of(conversation) -> Key:
    return Key(character=conversation.character, thread_id=conversation.identifier)


def _capture(envelope, key: Key, target_index: int, target_kind: str, target_version: int,
             accepted_revision, history, prefix: str, persisted: bool,
             opening: str = "", instruction=None, result_revision: int = 0) -> dict:
    """Freeze everything the request is built from, and build it. Once.

    Built here rather than on the executor thread so that a refusal the builder
    can raise -- a character that will not load, a vision requirement with no
    projector -- is a refusal the caller sees, not a reply that fails a second
    later with nobody watching.
    """
    from prompt_master.chat.characters import (DEFAULT_MAX_REPLY_TOKENS, DEFAULT_TEMPERATURE,
                                               DEFAULT_TOP_P)
    from prompt_master.chat.prompt import build, needs_vision
    from prompt_master.core.models import RANDOM_SEED, draw_seed

    import mc_llm_conversation_service as service
    import mc_llm_sessions as sessions

    character = _character(key.character)
    persona = _persona()
    settings = envelope.payload.get("settings") or {}

    tokens = _whole(settings.get("reply_tokens"), character.max_reply_tokens,
                    DEFAULT_MAX_REPLY_TOKENS)
    resolved = _whole(character.seed, RANDOM_SEED)
    if resolved == RANDOM_SEED:
        resolved = draw_seed()
    asked = _whole(settings.get("seed"), RANDOM_SEED)

    every = _every_picture()
    wire = build(character, persona, _with_pictures(list(history), every),
                 context_size=_context_size(), reply_tokens=tokens, instruction=instruction,
                 every_picture=every)
    request = sessions.ChatRequest(
        messages=wire,
        needs_vision=needs_vision(wire),
        temperature=_fraction(settings.get("temperature"), character.temperature,
                              DEFAULT_TEMPERATURE),
        top_p=_fraction(settings.get("top_p"), character.top_p, DEFAULT_TOP_P),
        max_tokens=tokens,
        seed=asked if asked != RANDOM_SEED else resolved,
    )
    if request.needs_vision and not _sees():
        raise service.Refused(service.VISION_UNAVAILABLE,
                              "The model running has no vision projector, so the attached "
                              "image cannot be sent to it.")
    return {"key": key, "target_index": target_index, "target_kind": target_kind,
            "target_version": target_version, "accepted_revision": accepted_revision,
            "opening": opening, "request": request, "character": character,
            "persona": persona, "persisted": persisted, "prefix": prefix,
            "result_revision": result_revision}


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _execute(operation: Operation, plan: dict) -> None:
    """The reply, on a thread of its own. Never raises into anything.

    Everything below is the panel's old ``_stream`` loop with the yields taken
    out and the saving moved to the end. The event stream, the cancellation
    token, the speech turn and the cleaning are the same calls in the same
    order; what changed is who is holding them.
    """
    import mc_llm_conversation_service as service
    import mc_llm_sessions as sessions

    opening = operation.opening
    streamed = opening
    join_space = bool(opening) and not opening[-1].isspace()
    last_checkpoint = 0.0
    failure = ""

    turn = operation.turn

    _advance(operation, WAITING)
    try:
        for event in sessions.conversation(operation.request, operation.cancel):
            if event.kind == sessions.CHUNK:
                if operation.phase != GENERATING:
                    _advance(operation, GENERATING)
                if join_space and event.text and not event.text[0].isspace():
                    streamed += " "
                join_space = False
                streamed += event.text
                if turn is not None:
                    # Non-blocking, always: the most this does is run a
                    # segmenter over a few hundred characters and put the
                    # result on a queue. It must never call an engine here.
                    try:
                        turn.add_text(event.text)
                    except Exception:
                        logger.debug("Model Chain: could not queue speech", exc_info=True)
                with _guard:
                    operation.text = streamed
                    operation.seq += 1
                    _guard.notify_all()
                _publish(operation, service.REPLY_PATCH, text=streamed,
                         target_index=operation.target_index, phase=operation.phase)
                now = time.monotonic()
                if now - last_checkpoint >= CHECKPOINT_INTERVAL:
                    last_checkpoint = now
                    checkpoint(operation, streamed)
            elif event.kind == sessions.STATUS:
                _advance(operation, PREPARING if operation.phase in (WAITING, ACCEPTED)
                         else operation.phase, status=event.text)
            elif event.kind in (sessions.DONE, sessions.CANCELLED):
                whole = event.text if event.kind == sessions.DONE and not opening else streamed
                streamed = whole or streamed
                _finish(operation, plan, streamed,
                        STOPPED if event.kind == sessions.CANCELLED else COMPLETED)
                return
            elif event.kind == sessions.FAILED:
                failure = event.text
                _finish(operation, plan, streamed, FAILED, error=failure)
                return
    except Exception as exc:
        logger.warning("Model Chain: a reply failed", exc_info=True)
        _finish(operation, plan, streamed, FAILED, error=str(exc))
        return

    # The event stream ended without a terminal event, which is the other way a
    # reply finishes whole: everything that arrived, arrived.
    _finish(operation, plan, streamed, COMPLETED)


def _finish(operation: Operation, plan: dict, streamed: str, outcome: str,
            error: str = "") -> None:
    """Clean the reply, write it where it belongs, and tell everybody. Once."""
    from prompt_master.chat.prompt import clean_reply

    import mc_llm_conversation_service as service

    character, persona = plan["character"], plan["persona"]
    turn = operation.turn
    if operation.stop_requested and outcome == COMPLETED:
        # Stop reached the token after the model had already finished. The text
        # is real and is kept; what does not happen is the automatic speech,
        # because Stop was accepted for a nonterminal operation (7.7).
        outcome = STOPPED
    try:
        whole = clean_reply(streamed or "", character, persona)
    except Exception:
        logger.debug("Model Chain: could not clean a reply", exc_info=True)
        whole = streamed or ""
    operation.text = whole

    if outcome == COMPLETED and turn is not None:
        try:
            turn.complete(_spoken_tail(whole, operation.opening))
        except Exception:
            logger.debug("Model Chain: could not finish the speech turn", exc_info=True)
    elif turn is not None:
        try:
            turn.cancel("stopped" if outcome == STOPPED else "failed")
        except Exception:
            logger.debug("Model Chain: could not cancel the speech turn", exc_info=True)

    if not whole.strip():
        # Nothing was generated. Nothing is written: the provisional reply was
        # never in the file, so "remove the provisional assistant and restore
        # the prior selected version" is simply the absence of a write.
        _release(operation)
        _advance(operation, outcome, error=error)
        _remember(operation, whole, outcome)
        return

    _advance(operation, SAVING)
    try:
        revision = _commit(operation, plan, whole)
    except store_module.StaleRevision as refusal:
        _keep(operation, whole, "result_conflict")
        _release(operation)
        operation.recovery = "result_conflict"
        _advance(operation, outcome, error="Your reply is ready, but the conversation "
                                           "changed.")
        _publish(operation, service.RESULT_CONFLICT, force=True, text=whole,
                 current_revision=getattr(refusal, "current", None))
        _remember(operation, whole, outcome)
        return
    except store_module.StoreRefusal as refusal:
        _keep(operation, whole, "save_failed")
        _release(operation)
        operation.recovery = "save_failed"
        _advance(operation, SAVE_FAILED, error=str(refusal))
        _remember(operation, whole, outcome)
        return

    operation.result_revision = revision
    _release(operation)
    _advance(operation, outcome, error=error)
    _remember(operation, whole, outcome)
    service.publish(service.CONVERSATION_CHANGED, operation.key, revision=revision)
    _forget_checkpoint(operation)


def reply_receipt(operation_id: str) -> str:
    """The receipt id one operation's *reply* is written under.

    Suffixed rather than shared, because an operation can write twice: the
    acceptance (a user turn, or a branch) and the completion. Both go into the
    same file's receipt list and a shared id would make the second look like a
    retry of the first.
    """
    return f"{operation_id}#reply"


def _commit(operation: Operation, plan: dict, whole: str) -> int:
    """Write the reply into the conversation it was accepted against. Guarded.

    ``expected`` is the revision at *acceptance*, not the revision now. That is
    the completion guard: if anything landed in between, this raises and the
    reply goes to recovery rather than over the top of whatever arrived.
    """
    import mc_llm_conversation_service as service

    store = service.chats()
    # A distinct receipt id for the *reply*, because an operation writes twice
    # and both writes land in a file that remembers operation ids. A branching
    # regenerate creates the branch under the command's own id; if the reply
    # were committed under that id too, the transaction would find the branch's
    # receipt, answer "already applied" and write nothing -- a reply silently
    # lost to its own operation's earlier step.
    receipt = Receipt(operation_id=reply_receipt(operation.operation_id),
                      action=operation.action, payload_digest="")

    def change(conversation):
        from prompt_master.chat.history import ASSISTANT

        index, kind = operation.target_index, operation.target_kind
        if kind == APPEND:
            if index != len(conversation.messages):
                raise store_module.StaleRevision(int(conversation.revision or 0))
            conversation.append(ASSISTANT, whole)
        elif kind == VERSION:
            if not 0 <= index < len(conversation.messages):
                raise store_module.StaleRevision(int(conversation.revision or 0))
            message = conversation.messages[index]
            if message.role != ASSISTANT or len(message.versions) != operation.target_version:
                raise store_module.StaleRevision(int(conversation.revision or 0))
            message.add_version(whole)
        else:
            if not 0 <= index < len(conversation.messages):
                raise store_module.StaleRevision(int(conversation.revision or 0))
            message = conversation.messages[index]
            if message.role != ASSISTANT \
                    or not 0 <= operation.target_version < len(message.versions):
                raise store_module.StaleRevision(int(conversation.revision or 0))
            message.active = operation.target_version
            message.text = whole
        conversation.retitle()
        return True

    def unchanged(conversation, raw, current):
        from prompt_master.chat.history import fingerprint

        if operation.accepted_fingerprint and fingerprint(raw) != operation.accepted_fingerprint:
            # Same revision, different bytes: something wrote this file without
            # going through the service. The reply is not written over it.
            raise store_module.StaleRevision(current)

    committed = store_module.transaction(store, operation.key, operation.accepted_revision,
                                         change, receipt=receipt, allow_busy=True,
                                         guard=unchanged)
    return committed.revision


def _remember(operation: Operation, whole: str, outcome: str) -> None:
    """Record what this operation produced, for whoever wants to speak it.

    Read by both views and consumed by neither (7.8). ``auto_speech_owner_page``
    is the page that asked, so exactly one window starts playing automatically
    and the others show "Playing in another window" with a Stop that works.
    """
    with _guard:
        _completions[operation.operation_id] = Completion(
            operation_id=operation.operation_id, character=operation.key.character,
            thread_id=operation.key.thread_id, final_text=whole,
            speech_turn_id=operation.speech_turn_id, status=outcome,
            auto_speech_owner_page=operation.page_id if outcome == COMPLETED
            and not operation.stop_requested else "")
        for key in [key for key, found in _completions.items()
                    if time.time() - found.at > RECOVERY_TTL]:
            _completions.pop(key, None)


def _release(operation: Operation) -> None:
    import mc_llm_conversation_service as service

    try:
        store_module.release(service.chats(), operation.key, operation.operation_id)
    except Exception:
        logger.debug("Model Chain: could not release the conversation claim", exc_info=True)


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #


def stop(operation_id: str) -> bool:
    """Cancel one operation. Harmless when repeated, or when it has finished.

    Sets the operation's own :class:`mc_llm_sessions.Cancellation`, which is
    exactly what the Stop button set before -- the difference is that the token
    now belongs to the operation rather than to a Gradio generator, so closing
    the generator does not cancel anything and Stop does not depend on one
    being open.
    """
    import mc_llm_conversation_service as service

    with _guard:
        operation = _operations.get(str(operation_id or ""))
        if operation is None:
            return False
        already = operation.phase in TERMINAL
        operation.stop_requested = True
    if already:
        _quiet(operation)
        return True
    try:
        operation.cancel.cancel()
    except Exception:
        logger.debug("Model Chain: could not cancel a reply", exc_info=True)
    _quiet(operation)
    _advance(operation, STOPPING)
    service.publish(service.SPEECH_STATE, operation.key,
                    operation_id=operation.operation_id, playing=False, reason="stopped")
    return True


def _quiet(operation: Operation) -> None:
    """Stop this operation's audio, wherever it had got to."""
    turn = operation.turn
    if turn is None:
        return
    try:
        if getattr(turn, "busy", False):
            turn.cancel("stopped")
    except Exception:
        logger.debug("Model Chain: could not stop this reply's audio", exc_info=True)


def stop_for(key: Key) -> bool:
    operation = for_conversation(key)
    return stop(operation.operation_id) if operation is not None else False


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def _ops_folder() -> Path | None:
    try:
        import mc_llm_paths

        folder = Path(mc_llm_paths.app_paths().chats) / ".v2" / "ops"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    except Exception:
        logger.debug("Model Chain: could not open the operations folder", exc_info=True)
        return None


def checkpoint(operation: Operation, text: str) -> bool:
    """Write down what a running reply holds. Never a conversation write.

    A checkpoint does not touch the chat file and does not move the revision,
    which is what makes it safe to take while another window is reading. What
    it buys is that a restart can show the person the reply that was in flight
    rather than nothing at all -- bounded by the interval, so at most one
    second of tokens is ever lost.
    """
    folder = _ops_folder()
    if folder is None:
        return False
    try:
        from prompt_master.core.config import atomic_write_json

        atomic_write_json(folder / f"{operation.operation_id}.json",
                          {"operation_id": operation.operation_id,
                           "character": operation.key.character,
                           "thread_id": operation.key.thread_id,
                           "action": operation.action,
                           "target_index": operation.target_index,
                           "target_kind": operation.target_kind,
                           "target_version": operation.target_version,
                           "accepted_revision": operation.accepted_revision,
                           "opening": operation.opening,
                           "text": text, "at": time.time()})
        return True
    except Exception:
        logger.debug("Model Chain: could not write a reply checkpoint", exc_info=True)
        return False


def _forget_checkpoint(operation: Operation) -> None:
    folder = _ops_folder()
    if folder is None:
        return
    try:
        (folder / f"{operation.operation_id}.json").unlink(missing_ok=True)
    except OSError:
        logger.debug("Model Chain: could not remove a reply checkpoint", exc_info=True)


def interrupted() -> list:
    """Checkpoints left by a process that did not finish. Never auto-resumed.

    Read at startup and offered; nothing here regenerates, resends or speaks
    anything. A reply the machine was in the middle of when it was restarted is
    a reply the person decides about.
    """
    folder = _ops_folder()
    if folder is None:
        return []
    found = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data["phase"] = INTERRUPTED
            data["status"] = PHASE_TEXT[INTERRUPTED]
            found.append(data)
    return found


def clear_interrupted(operation_id: str) -> None:
    folder = _ops_folder()
    if folder is None:
        return
    try:
        (folder / f"{str(operation_id or '')}.json").unlink(missing_ok=True)
    except OSError:
        logger.debug("Model Chain: could not clear an interrupted reply", exc_info=True)


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #


def _reserve_recovery() -> bool:
    """Whether there is room to preserve a result before work is accepted."""
    with _guard:
        if len(_unresolved) >= MAX_RECOVERY:
            return False
        return sum(item.size for item in _unresolved.values()) < MAX_RECOVERY_BYTES


def _keep(operation: Operation, text: str, reason: str) -> None:
    with _guard:
        _unresolved[operation.operation_id] = Unresolved(
            operation_id=operation.operation_id, character=operation.key.character,
            thread_id=operation.key.thread_id,
            accepted_revision=operation.accepted_revision, text=text,
            target_index=operation.target_index, target_kind=operation.target_kind,
            reason=reason)


def resolve(operation_id: str, decision: str) -> dict:
    """What the person chose for an unsaved result: discard, or a new branch.

    "Save as new branch" builds from the *captured* context and the result,
    never from the conversation as it is now -- the conversation is exactly
    what changed, and branching from it would put the reply under messages it
    was not answering.
    """
    import mc_llm_conversation_service as service

    with _guard:
        item = _unresolved.get(str(operation_id or ""))
    if item is None:
        raise service.Refused(service.NOT_FOUND, "There is no unsaved reply by that name.")
    if decision == "discard":
        with _guard:
            _unresolved.pop(item.operation_id, None)
        return {"ok": True, "discarded": True}
    if decision != "branch":
        raise service.Refused(service.INVALID_INPUT, "Choose save or discard.")

    from prompt_master.chat.history import ASSISTANT

    store = service.chats()
    key = Key(character=item.character, thread_id=item.thread_id)
    source, _ = store_module.read(store, key)
    index = min(max(item.target_index - 1, -1), len(source.messages) - 1)
    identifier = store.new(key.character).identifier
    made = Key(character=key.character, thread_id=identifier)

    def build():
        branched = source.branch(index, identifier)
        branched.append(ASSISTANT, item.text)
        return branched

    committed = store_module.create(store, made, build)
    with _guard:
        _unresolved.pop(item.operation_id, None)
    service.publish(service.THREAD_CREATED, made, revision=committed.revision)
    return {"ok": True, "resulting_conversation": {"character": made.character,
                                                   "thread_id": made.thread_id},
            "revision": committed.revision}


# --------------------------------------------------------------------------- #
# Helpers carried over from the panel, unchanged in behaviour
# --------------------------------------------------------------------------- #


def _character(who: str):
    from prompt_master.chat.characters import Character, CharacterStore

    import mc_llm_paths

    try:
        return CharacterStore.from_paths(mc_llm_paths.app_paths()).load(who) if who \
            else Character(name="")
    except Exception:
        logger.debug("Model Chain: could not load the character %s", who, exc_info=True)
        return Character(name=who or "")


def _persona():
    from prompt_master.chat.characters import load_persona

    import mc_llm_paths

    try:
        return load_persona(mc_llm_paths.app_paths())
    except Exception:
        from prompt_master.chat.characters import Persona

        logger.debug("Model Chain: could not read the persona", exc_info=True)
        return Persona()


def _sees() -> bool:
    try:
        import mc_llm_runtime

        return bool(mc_llm_runtime.config().sees)
    except Exception:
        logger.debug("Model Chain: could not read the vision configuration", exc_info=True)
        return True


def _context_size() -> int:
    import mc_llm_chat_panel

    return mc_llm_chat_panel._context_size()


def _with_pictures(messages, every_picture: bool = False):
    import mc_llm_chat_panel

    return mc_llm_chat_panel._with_pictures(messages, every_picture)


def _every_picture() -> bool:
    """The *Show the model every picture* setting, read at the moment of the reply."""
    import mc_llm_vision

    return mc_llm_vision.every_picture()


def _spoken_tail(whole: str, opening: str) -> str:
    import mc_llm_chat_panel

    return mc_llm_chat_panel._spoken_tail(whole, opening)


def _last_reply(conversation, index) -> int:
    from prompt_master.chat.history import ASSISTANT

    try:
        index = int(index)
    except (TypeError, ValueError):
        index = -1
    if index >= 0:
        return index
    return conversation.last_index(ASSISTANT)


def _whole(value, *fallbacks) -> int:
    for candidate in (value,) + fallbacks:
        try:
            return int(float(candidate))
        except (TypeError, ValueError):
            continue
    return 0


def _fraction(value, *fallbacks) -> float:
    for candidate in (value,) + fallbacks:
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return 0.0
