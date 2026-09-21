"""The one place a conversation is written. Locks, revisions, receipts.

Two windows onto one chat is the whole problem this module exists for. Before
it, ``ChatStore.save()`` was ``atomic_write_json`` and nothing else: the write
could not be torn in half, and it could not be stopped from being the wrong
write. A second page that had loaded the thread ten seconds earlier, edited a
message and pressed Save, saved *its* copy of the thread — every reply that had
arrived in between, gone, with no error anywhere and nothing on screen to
suggest anything had happened.

The fix is not a bigger lock. It is a *comparison*, and the lock is only what
makes the comparison mean something:

    read the file, work out what it compares as, refuse if that is not what the
    caller last saw, mutate a copy, write the copy, remember the new number.

All of it inside one lock per conversation, and none of it anywhere near
inference. The critical section is a file read, a dictionary copy and a file
write; a model is never loaded inside one, a socket is never written inside
one, and nothing here ever waits for a card.

What a revision is
------------------
An integer on the conversation, bumped by one on every committed write here.
Nowhere else bumps it — not token progress, not a checkpoint, not a write that
failed. A caller sends back the number it last saw and gets
:class:`StaleRevision` if the file has moved on.

A chat written before this existed has no number, and reading one must not
write one: reading is how the list of threads is drawn, and a store that wrote
on read would rewrite every file in the folder the first time somebody opened
the thread list. So a revision-0 file compares by the SHA-256 of its bytes
instead, which is a fact about what the caller saw that needs nothing written
down. The first guarded write makes it revision 1 and the fingerprint is never
needed again for that file.

What an active writer is
------------------------
A generation owns its conversation from the moment it is accepted until it is
saved or abandoned. While it does, a competing *content* mutation gets
:class:`ThreadBusy` rather than being queued behind it — because the thing it
would be queued behind is a language model, and a queue whose head is
"whenever the model finishes" is a UI that has silently hung. Reading,
browsing, selecting, drafting and Stop are all still available; nothing waits
in silence.

Receipts, and what they are for
-------------------------------
Every committed transaction appends a small record of itself to the file. It is
how a retry after a dropped acknowledgement is answered without doing the work
twice: the operation id is in the file, so the answer is "already applied"
rather than a second copy of the user's message. Sixty-four per file, which is
far more than any retry needs and small enough to be invisible beside the
messages.

Durability is tier 1 (specification 5.7): single-file writes are atomic, and a
multi-file operation — a branch is two files — can leave an orphan after a
crash but never a corrupt file. The startup scan below reports orphans; it does
not delete them, because an orphan is somebody's conversation.
"""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

MAX_RECEIPTS = 64
"""How many receipts one chat file keeps. Oldest dropped first.

A retry happens within seconds of the request it repeats, so one would very
nearly do. Sixty-four is room for a page that lost its connection during a
burst of edits, and about two kilobytes of JSON.
"""

LOCK_WAIT = 30.0
"""How long a transaction waits for the conversation's lock before refusing.

Long enough that no honest critical section here -- a read, a copy, a write --
can be behind it, and short enough that a deadlock that should be impossible
surfaces as a refusal somebody reports rather than as a tab that never answers.
"""


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


class StoreRefusal(Exception):
    """Base for everything this module refuses to do. Carries a code."""

    code = "SAVE_FAILED"

    def __init__(self, message: str = "", **detail):
        super().__init__(message or self.__doc__ or self.code)
        self.detail = detail


class StaleRevision(StoreRefusal):
    """This conversation changed in another window. Your draft is safe."""

    code = "STALE_REVISION"

    def __init__(self, current=None, message: str = ""):
        super().__init__(message, current_revision=current)
        self.current = current


class ThreadBusy(StoreRefusal):
    """Reply in progress. Stop it before changing this conversation."""

    code = "THREAD_BUSY"

    def __init__(self, operation_id: str = "", message: str = ""):
        super().__init__(message, operation_id=operation_id)
        self.operation_id = operation_id


class MetadataDamaged(StoreRefusal):
    """Conversation metadata needs repair."""

    code = "STORAGE_EXTERNALLY_CHANGED"


class ConversationMissing(StoreRefusal):
    """This conversation was deleted."""

    code = "NOT_FOUND"


class SaveFailed(StoreRefusal):
    """Response not saved."""

    code = "SAVE_FAILED"


class StorageChanged(StoreRefusal):
    """This conversation was changed outside the application."""

    code = "STORAGE_EXTERNALLY_CHANGED"


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Key:
    """Which conversation. A character and a thread id, and nothing else.

    Frozen so it can be a dictionary key, and compared by value so two pages
    naming the same thread name the same lock. The *path* is derived rather
    than carried: a key that held a path would be a key a browser could put a
    path into.
    """

    character: str
    thread_id: str

    def __str__(self) -> str:                       # pragma: no cover - display
        return f"{self.character}/{self.thread_id}"

    @property
    def valid(self) -> bool:
        return bool(str(self.character or "").strip() and str(self.thread_id or "").strip())


def key_of(conversation) -> Key:
    return Key(character=str(getattr(conversation, "character", "") or ""),
               thread_id=str(getattr(conversation, "identifier", "") or ""))


def canonical(store, key: Key) -> Path:
    """Where this key's file is, resolved, and proven to be inside the store.

    The check is not decoration. ``character`` and ``thread_id`` arrive from a
    browser, and ``safe_stem`` alone is a filter rather than a proof --
    ``resolve()`` and a prefix test is the proof, and it is applied to every
    key before anything is opened. A key that escapes the chats folder is not a
    key with an unusual name; it is an attempt to read somebody's ssh config.
    """
    if not key.valid:
        raise ConversationMissing("Choose a character and a thread first.")
    root = Path(store.directory).resolve()
    path = Path(store.path_for(key.character, key.thread_id))
    try:
        resolved = path.resolve()
    except OSError:                                  # pragma: no cover - exotic
        raise ConversationMissing("Choose a character and a thread first.")
    if root not in resolved.parents:
        raise ConversationMissing("Choose a character and a thread first.")
    return resolved


# --------------------------------------------------------------------------- #
# The lock registry
# --------------------------------------------------------------------------- #


class _Registry:
    """One lock per conversation, and what is known about each right now.

    A module-level singleton, which is the point of it: a second
    ``ChatStore`` instance -- and this repository makes one per call, from
    ``_chats()`` -- must not come with a second lock, or two callers holding
    "the" lock would be holding two different objects and neither would be
    waiting for the other.

    Keyed by the resolved path rather than by the key, so two spellings of one
    character cannot be two locks on one file.
    """

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        self._revisions: dict[str, object] = {}
        self._writers: dict[str, dict] = {}

    # -- locks ------------------------------------------------------------ #

    def lock_for(self, path: Path) -> threading.RLock:
        name = str(path)
        with self._guard:
            found = self._locks.get(name)
            if found is None:
                found = self._locks[name] = threading.RLock()
            return found

    def acquire(self, paths) -> list:
        """Locks for several conversations, taken in a fixed order.

        Sorted by path, always, which is the whole of the deadlock story: a
        branch takes two locks and a delete takes two, and two operations
        taking the same two in opposite orders is the textbook way to stop a
        process dead. Sorting by something neither caller chooses makes the
        order a property of the files rather than of the code path.
        """
        ordered = sorted({str(path) for path in paths})
        taken = []
        try:
            for name in ordered:
                lock = self.lock_for(Path(name))
                if not lock.acquire(timeout=LOCK_WAIT):
                    raise SaveFailed("This conversation is busy. Try again.")
                taken.append(lock)
        except BaseException:
            for lock in reversed(taken):
                lock.release()
            raise
        return taken

    @staticmethod
    def release(taken) -> None:
        for lock in reversed(taken):
            try:
                lock.release()
            except RuntimeError:                     # pragma: no cover - defensive
                logger.debug("Model Chain: a conversation lock was released twice",
                             exc_info=True)

    # -- what is known ---------------------------------------------------- #

    def note_committed(self, path: Path, revision) -> None:
        with self._guard:
            self._revisions[str(path)] = revision

    def known_revision(self, path: Path):
        with self._guard:
            return self._revisions.get(str(path))

    def forget(self, path: Path) -> None:
        with self._guard:
            self._revisions.pop(str(path), None)
            self._writers.pop(str(path), None)

    # -- active writers --------------------------------------------------- #

    def claim(self, path: Path, operation_id: str, detail: dict | None = None) -> None:
        with self._guard:
            self._writers[str(path)] = {"operation_id": str(operation_id),
                                        "since": time.time(), **(detail or {})}

    def active_writer(self, path: Path) -> dict | None:
        with self._guard:
            return dict(self._writers[str(path)]) if str(path) in self._writers else None

    def release_claim(self, path: Path, operation_id: str = "") -> None:
        with self._guard:
            found = self._writers.get(str(path))
            if found is None:
                return
            if operation_id and found.get("operation_id") != str(operation_id):
                return
            self._writers.pop(str(path), None)

    def writers(self) -> dict:
        with self._guard:
            return {name: dict(detail) for name, detail in self._writers.items()}

    def reset(self) -> None:
        """Drop everything. For the tests, and for a UI reload."""
        with self._guard:
            self._locks.clear()
            self._revisions.clear()
            self._writers.clear()


registry = _Registry()
"""The process-wide singleton. There is deliberately no way to make a second."""


# --------------------------------------------------------------------------- #
# Receipts
# --------------------------------------------------------------------------- #


@dataclass
class Receipt:
    """What one committed transaction says about itself, afterwards."""

    operation_id: str
    action: str
    issued_at: str = ""
    payload_digest: str = ""
    accepted_revision: object = None
    result_revision: int = 0
    resulting_conversation: str = ""

    def summary(self) -> dict:
        found = {"op_id": self.operation_id, "action": self.action,
                 "issued_at": self.issued_at, "payload_digest": self.payload_digest,
                 "accepted_revision": self.accepted_revision,
                 "result_revision": self.result_revision}
        if self.resulting_conversation:
            found["resulting_conversation"] = self.resulting_conversation
        return found


def receipts_of(conversation) -> list:
    found = (getattr(conversation, "extra", None) or {}).get("receipts")
    return [row for row in found if isinstance(row, dict)] if isinstance(found, list) else []


def receipt_for(conversation, operation_id: str) -> dict | None:
    """This operation's receipt in this file, if it has one.

    What makes a retry after a restart safe: the registry that would have
    remembered the outcome is gone with the process, and the file is not.
    """
    wanted = str(operation_id or "")
    for row in reversed(receipts_of(conversation)):
        if str(row.get("op_id") or "") == wanted:
            return row
    return None


def _append_receipt(conversation, receipt: Receipt | None) -> None:
    if receipt is None:
        return
    extra = getattr(conversation, "extra", None)
    if not isinstance(extra, dict):
        extra = conversation.extra = {}
    kept = [row for row in receipts_of(conversation)
            if str(row.get("op_id") or "") != receipt.operation_id]
    kept.append(receipt.summary())
    extra["receipts"] = kept[-MAX_RECEIPTS:]


# --------------------------------------------------------------------------- #
# The transaction
# --------------------------------------------------------------------------- #


UNCHECKED = object()
"""``expected`` for a caller that is deliberately not comparing.

Used by maintenance that has just read the file inside the same lock, and by
nothing a browser can reach.
"""


def token_of(conversation, raw: bytes) -> object:
    """What this conversation compares as: its revision, or its bytes."""
    from prompt_master.chat.history import fingerprint

    revision = int(getattr(conversation, "revision", 0) or 0)
    return revision if revision else fingerprint(raw)


def matches(expected, token) -> bool:
    """Whether a client's ``expected`` names the token a file carries now.

    Deliberately not ``==``. A client that has never seen the file sends
    ``None``, which matches nothing; a client that saw revision 0 sends the
    fingerprint string; a client that saw a revision sends the integer. Booleans
    are excluded for the same reason they are in the file: ``True == 1``.
    """
    if expected is UNCHECKED:
        return True
    if isinstance(expected, bool) or isinstance(token, bool):
        return False
    if isinstance(expected, int) and isinstance(token, int):
        return expected == token
    if isinstance(expected, str) and isinstance(token, str):
        return expected == token
    return False


@dataclass
class Committed:
    """What a transaction leaves behind: the result, and where it landed."""

    result: object = None
    revision: int = 0
    conversation: object = None
    written: bool = False
    duplicate: bool = False
    receipt: dict = field(default_factory=dict)


def read(store, key: Key):
    """One conversation, read. Never writes, ever, for any reason.

    The second clause is load-bearing rather than obvious. The path this
    replaces adopted inline attachments and *saved* during a read, so drawing a
    thread list could rewrite files. Adoption still happens -- see
    :func:`maintain` -- but as a transaction somebody asked for, not as a side
    effect of looking.
    """
    from prompt_master.chat.history import Conversation

    path = canonical(store, key)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ConversationMissing("This conversation was deleted.")
    except OSError as exc:
        raise SaveFailed(f"Could not read this conversation: {exc}")
    conversation = _parse(raw)
    if conversation is None:
        raise StorageChanged("This conversation was changed outside the application.")
    if not isinstance(conversation, Conversation):                 # pragma: no cover
        raise StorageChanged("This conversation was changed outside the application.")
    if not conversation.damaged:
        conversation.comparison = token_of(conversation, raw)
    return conversation, raw


def _parse(raw: bytes):
    import json

    from prompt_master.chat.history import Conversation

    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return Conversation.from_dict(data)


def token(store, key: Key) -> object:
    """This conversation's comparison token, read fresh. Never writes."""
    conversation, raw = read(store, key)
    if conversation.damaged:
        raise MetadataDamaged()
    return token_of(conversation, raw)


def transaction(store, key: Key, expected, change, receipt: Receipt | None = None,
                also: list | None = None, allow_busy: bool = False,
                guard=None) -> Committed:
    """Read, compare, mutate a copy, write it, remember the number. Under lock.

    ``change`` is handed a *copy* of the conversation and may raise: nothing has
    been written when it does, because the copy is what it was given. A change
    that decides there is nothing to do returns ``False`` and the file is left
    exactly as it is -- including its revision, which must not move for a write
    that did not happen.

    ``also`` names other conversations this operation touches, so their locks
    are taken in the same sorted order (see :meth:`_Registry.acquire`). A branch
    is the case: it reads one file and writes another, and two branches crossing
    in opposite directions must not be able to stop each other.

    ``allow_busy`` is for the operation that *is* the active writer saving its
    own reply. Everything else is refused while a generation owns the thread.

    ``guard`` is a second opinion, given the conversation, its exact bytes and
    the token they compare as, and free to raise. It exists for the one case the
    revision cannot see: a revision only moves when *this* code writes, so a
    file changed by something outside the application -- an editor, a sync
    client, a second process -- comes back with different bytes and the same
    number. A generation that has been away for a minute is exactly where that
    matters, so its completion compares the bytes as well.
    """
    path = canonical(store, key)
    others = [canonical(store, other) for other in (also or [])]
    taken = registry.acquire([path] + others)
    try:
        conversation, raw = read(store, key)
        if conversation.damaged:
            raise MetadataDamaged()
        current = token_of(conversation, raw)
        if not matches(expected, current):
            raise StaleRevision(current)
        if receipt is not None:
            known = receipt_for(conversation, receipt.operation_id)
            if known is not None:
                # Already applied, on a previous attempt whose answer never
                # reached the caller. The work is not done again and the
                # recorded outcome is what comes back.
                return Committed(result=None, revision=int(conversation.revision or 0),
                                 conversation=conversation, written=False,
                                 duplicate=True, receipt=known)
        owner = registry.active_writer(path)
        if owner and not allow_busy and (receipt is None
                                         or owner.get("operation_id") != receipt.operation_id):
            raise ThreadBusy(str(owner.get("operation_id") or ""))
        if guard is not None:
            guard(conversation, raw, current)

        copy = _copy(conversation)
        result = change(copy)
        if result is False:
            return Committed(result=result, revision=int(conversation.revision or 0),
                             conversation=conversation, written=False)
        copy.revision = int(conversation.revision or 0) + 1
        if receipt is not None:
            receipt.accepted_revision = current
            receipt.result_revision = copy.revision
            _append_receipt(copy, receipt)
        try:
            store.save(copy, expected=current)
        except Exception as exc:
            from prompt_master.chat.history import RevisionMismatch

            if isinstance(exc, RevisionMismatch):
                raise StorageChanged("This conversation was changed outside the application.")
            logger.warning("Model Chain: could not save this conversation", exc_info=True)
            raise SaveFailed(f"Response not saved: {exc}")
        registry.note_committed(path, copy.revision)
        return Committed(result=result, revision=copy.revision, conversation=copy,
                         written=True,
                         receipt=receipt.summary() if receipt is not None else {})
    finally:
        registry.release(taken)


def _copy(conversation):
    """A deep copy through the store's own serialisation, not ``deepcopy``.

    Through ``to_dict``/``from_dict`` because that is the round trip the file
    will make anyway: a copy made any other way could hold something that does
    not survive being written, and the first anybody would know of it is the
    next time the thread was opened.
    """
    from prompt_master.chat.history import Conversation

    copy = Conversation.from_dict(conversation.to_dict())
    copy.revision = int(getattr(conversation, "revision", 0) or 0)
    copy.damaged = bool(getattr(conversation, "damaged", False))
    return copy


def create(store, key: Key, build, receipt: Receipt | None = None) -> Committed:
    """Write a conversation that does not exist yet. Revision 1.

    Separate from :func:`transaction` because there is nothing to compare
    against: the identifier was reserved exclusively (see
    ``ChatStore._identifier``), so the only question is whether this process is
    about to write over a file it reserved, and the answer is that nobody else
    could have been given that name.
    """
    path = canonical(store, key)
    if store.entombed(key.character, key.thread_id):
        raise ConversationMissing("This conversation was deleted.")
    taken = registry.acquire([path])
    try:
        conversation = build()
        conversation.revision = 1
        if receipt is not None:
            receipt.accepted_revision = None
            receipt.result_revision = 1
            _append_receipt(conversation, receipt)
        try:
            store.save(conversation)
        except Exception as exc:
            logger.warning("Model Chain: could not save a new conversation", exc_info=True)
            raise SaveFailed(f"Response not saved: {exc}")
        registry.note_committed(path, 1)
        return Committed(result=conversation, revision=1, conversation=conversation,
                         written=True,
                         receipt=receipt.summary() if receipt is not None else {})
    finally:
        registry.release(taken)


def remove(store, key: Key, expected, receipt: Receipt | None = None) -> Committed:
    """Delete a conversation, under the same comparison every write gets.

    A deletion is a write. It was not guarded before, which meant a thread
    could be deleted out from under a reply that was still arriving in it, and
    the reply's save would then recreate the file with one message in it.
    """
    path = canonical(store, key)
    taken = registry.acquire([path])
    try:
        conversation, raw = read(store, key)
        if not conversation.damaged:
            current = token_of(conversation, raw)
            if not matches(expected, current):
                raise StaleRevision(current)
        owner = registry.active_writer(path)
        if owner:
            raise ThreadBusy(str(owner.get("operation_id") or ""))
        store.delete(key.character, key.thread_id)
        registry.forget(path)
        return Committed(result=True, revision=0, conversation=None, written=True,
                         receipt=receipt.summary() if receipt is not None else {})
    finally:
        registry.release(taken)


def maintain(store, key: Key, change) -> Committed:
    """A write nobody asked for, taken only when it is genuinely needed.

    Attachment adoption, and only that so far. It is a transaction like any
    other -- same lock, same revision bump -- with two differences that matter:
    it compares against whatever it finds rather than against what a client
    saw, because no client asked for it, and it does nothing at all while an
    operation owns the thread, because a background tidy-up must never be the
    reason a reply cannot be saved.
    """
    path = canonical(store, key)
    if registry.active_writer(path):
        return Committed(result=False, written=False)
    taken = registry.acquire([path])
    try:
        conversation, raw = read(store, key)
        if conversation.damaged:
            raise MetadataDamaged()
        if registry.active_writer(path):
            return Committed(result=False, revision=int(conversation.revision or 0),
                             conversation=conversation, written=False)
        copy = _copy(conversation)
        if change(copy) is False:
            return Committed(result=False, revision=int(conversation.revision or 0),
                             conversation=conversation, written=False)
        copy.revision = int(conversation.revision or 0) + 1
        store.save(copy, expected=token_of(conversation, raw))
        registry.note_committed(path, copy.revision)
        return Committed(result=True, revision=copy.revision, conversation=copy, written=True)
    finally:
        registry.release(taken)


# --------------------------------------------------------------------------- #
# Active writers
# --------------------------------------------------------------------------- #


def claim(store, key: Key, operation_id: str, detail: dict | None = None) -> None:
    registry.claim(canonical(store, key), operation_id, detail)


def release(store, key: Key, operation_id: str = "") -> None:
    registry.release_claim(canonical(store, key), operation_id)


def active_writer(store, key: Key) -> dict | None:
    return registry.active_writer(canonical(store, key))


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #


def orphans(store, character: str) -> list:
    """Chat files that parse but hold nothing, from an interrupted create.

    Tier 1's one accepted imperfection (specification 5.7): a crash between
    reserving a name and writing the conversation leaves an empty file. It is
    never a *corrupt* conversation -- there was nothing in it -- and it is
    reported rather than deleted, because this function cannot tell an
    interrupted create from a chat somebody is in the middle of importing.
    """
    found = []
    try:
        listing = sorted(store.folder(character).glob("*.json"))
    except OSError:
        return found
    for path in listing:
        try:
            if path.stat().st_size == 0:
                found.append(path)
        except OSError:                              # pragma: no cover - raced away
            continue
    return found


def report_orphans(store) -> list:
    """Every empty chat file under the store, logged once at startup."""
    found = []
    try:
        folders = [item for item in Path(store.directory).iterdir() if item.is_dir()]
    except OSError:
        return found
    for folder in folders:
        found.extend(orphans(store, folder.name))
    if found:
        logger.info("Model Chain: %d conversation file(s) were reserved and never written; "
                    "they are left where they are: %s", len(found),
                    ", ".join(str(path.name) for path in found[:8]))
    return found


def fsync_directory(path: Path) -> bool:
    """Ask the filesystem to make a rename durable. Never raises.

    ``atomic_write_json`` fsyncs the file it wrote and then renames it, which
    is atomic on every filesystem this runs on. Whether the *rename* has
    reached the disk is a separate question, and the answer on Linux is to
    fsync the directory. Not available on Windows, where it is also not needed
    for the same reason -- so a failure here is logged at debug and ignored.
    """
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EISDIR, errno.EINVAL, errno.ENOENT):
            logger.debug("Model Chain: could not open %s to flush it", path, exc_info=True)
        return False
    try:
        os.fsync(handle)
        return True
    except OSError:
        logger.debug("Model Chain: could not flush %s", path, exc_info=True)
        return False
    finally:
        os.close(handle)
