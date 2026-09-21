"""Chat history: the messages, the versions of each, and the chats themselves.

Two ideas hold everything the chat view can do.

*A message keeps its versions.* Regenerating a reply does not overwrite it — it
appends, and the message remembers which version is showing. That is what makes
"redo" reversible: the reply you liked two attempts ago is still there to page
back to, and only the showing version is ever sent to the model or saved as the
conversation's text.

*A branch is a copy.* Branching at a message writes a new chat holding
everything up to that point, which is oobabooga's own semantics and the only
kind that survives being reloaded: the two conversations then diverge as
ordinary chats with no shared state, rather than as a tree that every operation
afterwards has to understand.

Chats are JSON under ``<install root>/chats/<character>/``, one file each,
written through ``atomic_write_json`` so a crash mid-save cannot leave a
half-written conversation behind.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from prompt_master.core.config import atomic_write_json, read_json

from .characters import safe_stem

USER = "user"
ASSISTANT = "assistant"

# What a chat is called before anything has been said in it.
UNTITLED = "New chat"

# How much of the first message becomes the chat's name.
TITLE_LENGTH = 48


@dataclass
class Message:
    """One turn, and every text it has had."""

    role: str
    versions: list[str] = field(default_factory=lambda: [""])
    active: int = 0
    image_path: str = ""
    """Where this turn's still is kept, relative to the attachment folder.

    A reference and not the bytes, which is the opposite of what this held
    before and for a reason that only appears in use: the transcript is
    re-sent to the browser on every token of a reply, so a photograph inside a
    message is re-sent on every token. The picture lives in a folder beside
    the chats -- see :mod:`mc_llm_attachments` -- and what travels is a path.

    Still not a reference to *the user's own file*, which is what the note
    below was guarding against. The bytes are copied into a folder this
    application owns, so moving or deleting the original changes nothing.
    """
    image: str = ""
    """The still as an inline data URL: a chat written before there was a folder.

    Kept so those conversations keep working exactly as they did, and moved
    onto disk the first time one is opened -- see
    :func:`mc_llm_attachments.adopt`. A message never has both.
    """
    image_name: str = ""

    @property
    def text(self) -> str:
        return self.versions[self.active] if self.versions else ""

    @text.setter
    def text(self, value: str) -> None:
        if not self.versions:
            self.versions = [""]
        self.versions[self.active] = value

    def add_version(self, text: str = "") -> None:
        """Start a new version and show it — what a regenerate produces."""
        self.versions.append(text)
        self.active = len(self.versions) - 1

    def drop_version(self) -> None:
        """Discard the showing version, unless it is the only one."""
        if len(self.versions) > 1:
            self.versions.pop(self.active)
            self.active = min(self.active, len(self.versions) - 1)

    def show(self, index: int) -> None:
        if self.versions:
            self.active = max(0, min(index, len(self.versions) - 1))

    @property
    def attached(self) -> bool:
        """Whether this turn carries a still, wherever it is being kept."""
        return bool(self.image_path or self.image)

    def to_dict(self) -> dict:
        written = {"role": self.role, "versions": list(self.versions), "active": self.active,
                   "image_name": self.image_name}
        # One or the other and never both. Writing the inline copy beside the
        # path would keep every migrated chat exactly as large as it was, which
        # is the thing moving the pictures out was for.
        if self.image_path:
            written["image_path"] = self.image_path
        elif self.image:
            written["image"] = self.image
        return written

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        versions = data.get("versions")
        if not isinstance(versions, list) or not versions:
            # An older or hand-written file may carry a single "content".
            versions = [str(data.get("content", ""))]
        versions = [str(version) for version in versions]
        active = data.get("active", 0)
        active = active if isinstance(active, int) and 0 <= active < len(versions) else 0
        role = ASSISTANT if str(data.get("role")) == ASSISTANT else USER
        return cls(role=role, versions=versions, active=active,
                   image_path=str(data.get("image_path", "")),
                   image=str(data.get("image", "")),
                   image_name=str(data.get("image_name", "")))


def _revision_of(data: dict) -> tuple[int, bool]:
    """``(revision, damaged)`` for one parsed chat file.

    Absent is 0 and is not damage: that is every chat written before revisions
    existed. Present and not a nonnegative integer *is* damage, and is reported
    rather than corrected -- a counter reset to zero by a well-meaning reader is
    a counter two windows can then both win against.

    ``bool`` is excluded explicitly because ``True`` is an ``int`` in Python and
    would otherwise read as revision 1, which is the one wrong answer that looks
    entirely plausible in a file listing.
    """
    if "revision" not in data:
        return 0, False
    value = data.get("revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0, True
    return value, False


def fingerprint(raw: bytes) -> str:
    """The comparison token for a file that has no revision yet.

    A revision-0 file is one nothing guarded has ever written, so there is no
    counter to compare -- but the bytes on disk are still a fact about what the
    client last saw. SHA-256 of them is that fact in a form a browser can hold
    and hand back. It stops being needed the moment the first guarded write
    lands, because that write makes the file revision 1.
    """
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class RevisionMismatch(Exception):
    """A guarded write whose file no longer carries the token it expected."""

    def __init__(self, current, message: str = ""):
        super().__init__(message or "This conversation changed in another window.")
        self.current = current


@dataclass
class Conversation:
    identifier: str
    character: str
    title: str = UNTITLED
    created: float = 0.0
    updated: float = 0.0
    messages: list[Message] = field(default_factory=list)
    # The opening every reply in this chat is made to start with. It belongs to
    # the chat rather than to the character or to the app: two conversations
    # with the same character are two different things to be steering, and a
    # start written in one of them has no business turning up in the other.
    # Nothing but writing over it takes it away — not sending, not regenerating,
    # not reopening the chat a week later.
    response_prefix: str = ""

    revision: int = 0
    """How many guarded writes this conversation has had. Zero means none.

    The comparison token every mutation is validated against -- see
    :mod:`mc_llm_conversation_store`. A file written before there was such a
    thing reads as 0 and is *not* rewritten to say so: reading never writes,
    and the first guarded write is what makes it 1. A conversation that has
    never been through the guarded store therefore looks exactly as it always
    did, which is what makes an older copy of this extension able to open a
    file a newer one wrote.
    """

    extra: dict = field(default_factory=dict)
    """The ``v2`` namespace: receipts, and anything later versions add.

    Kept out of the named fields because it is not conversation content. It
    round-trips so that a write by this version does not drop what another
    wrote, and :meth:`from_dict` already tolerates unknown keys, so an older
    extension reading one of these files simply drops it on the next save --
    which is why the release notes say downgrade is not write-interoperable.
    """

    comparison: object = None
    """What this copy compares as, if it was read through the guarded store.

    A revision integer once anything guarded has written the file, the byte
    fingerprint before that, and ``None`` for a conversation that was built
    rather than read. Not serialised: it is a fact about the *read*, and
    writing it into the file would be writing down what the file used to be.

    It is here so that a panel which has just loaded a thread can say which
    revision the person is looking at without reading the file a second time --
    and every mutation from that panel carries it.
    """

    damaged: bool = False
    """Whether this file's revision could not be read as a revision.

    Not serialised, and deliberately not a refusal to *load*: a conversation
    whose metadata is wrong is still a conversation somebody wants to read.
    What it is, is a conversation nothing may write -- see
    :func:`mc_llm_conversation_store.transaction`, which raises rather than
    silently resetting a counter it does not understand.
    """

    # ── editing ──────────────────────────────────────────────────────────────

    def append(self, role: str, text: str = "", image: str = "", image_name: str = "",
               image_path: str = "") -> Message:
        message = Message(role=role, versions=[text], image=image, image_name=image_name,
                          image_path=image_path)
        self.messages.append(message)
        self.retitle()
        return message

    def delete(self, index: int) -> None:
        if 0 <= index < len(self.messages):
            del self.messages[index]

    def delete_from(self, index: int) -> None:
        """This message and everything after it."""
        if 0 <= index < len(self.messages):
            del self.messages[index:]

    def truncate_after(self, index: int) -> None:
        """Everything after this message — what regenerating from it needs."""
        if 0 <= index < len(self.messages):
            del self.messages[index + 1:]

    def last_index(self, role: str | None = None) -> int:
        for index in range(len(self.messages) - 1, -1, -1):
            if role is None or self.messages[index].role == role:
                return index
        return -1

    def retitle(self) -> None:
        """Name an untitled chat after the first thing said in it."""
        if self.title != UNTITLED:
            return
        for message in self.messages:
            if message.role == USER and message.text.strip():
                line = " ".join(message.text.split())
                self.title = line[:TITLE_LENGTH].rstrip() + ("…" if len(line) > TITLE_LENGTH else "")
                return

    def branch(self, index: int, identifier: str) -> "Conversation":
        """A new chat holding everything up to and including ``index``."""
        now = time.time()
        kept = [Message.from_dict(message.to_dict()) for message in self.messages[:index + 1]]
        title = self.title if self.title != UNTITLED else UNTITLED
        return Conversation(identifier=identifier, character=self.character,
                            title=f"{title} (branch)" if title != UNTITLED else UNTITLED,
                            created=now, updated=now, messages=kept,
                            # A branch carries on from here, and the start the
                            # replies were being given is part of what "here" is.
                            response_prefix=self.response_prefix)

    # ── storage ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        written = {"id": self.identifier, "character": self.character, "title": self.title,
                   "created": self.created, "updated": self.updated,
                   "response_prefix": self.response_prefix,
                   "messages": [message.to_dict() for message in self.messages]}
        # Written only once there is one. A chat that has never been through
        # the guarded store keeps the exact shape it had, so upgrading this
        # extension does not rewrite every file in the folder.
        if self.revision:
            written["revision"] = int(self.revision)
        if self.extra:
            written["v2"] = self.extra
        return written

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        messages = data.get("messages")
        revision, damaged = _revision_of(data)
        extra = data.get("v2")
        return cls(
            revision=revision,
            damaged=damaged,
            extra=dict(extra) if isinstance(extra, dict) else {},
            identifier=str(data.get("id", "")),
            character=str(data.get("character", "")),
            title=str(data.get("title") or UNTITLED),
            created=float(data.get("created", 0.0) or 0.0),
            updated=float(data.get("updated", 0.0) or 0.0),
            # A chat written before there was such a thing simply has none.
            response_prefix=str(data.get("response_prefix") or ""),
            messages=[Message.from_dict(item) for item in messages if isinstance(item, dict)]
            if isinstance(messages, list) else [],
        )


@dataclass(frozen=True)
class ChatInfo:
    """One row of the past-chats list."""

    identifier: str
    title: str
    updated: float


class ChatStore:
    """Every saved conversation, filed under the character it belongs to."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @classmethod
    def from_paths(cls, paths) -> "ChatStore":
        return cls(paths.chats)

    def folder(self, character: str) -> Path:
        return self.directory / safe_stem(character or "unnamed")

    def path_for(self, character: str, identifier: str) -> Path:
        return self.folder(character) / f"{safe_stem(identifier)}.json"

    def new(self, character: str) -> Conversation:
        now = time.time()
        return Conversation(identifier=self._identifier(character), character=character,
                            created=now, updated=now)

    UNCHECKED = object()
    """What ``expected`` holds when a caller is not comparing anything.

    The default, and therefore the behaviour every existing caller keeps: last
    writer wins. It is named rather than ``None`` because ``None`` is a real
    token -- it is what a caller that has never seen this file would send.
    """

    def save(self, conversation: Conversation, expected=UNCHECKED) -> Path:
        """Write one conversation. With ``expected``, only if it has not moved.

        The comparison is the guarded store's (see
        :mod:`mc_llm_conversation_store`), and lives here because this is the
        only function that knows both the path and the bytes about to replace
        what is at it. ``expected`` is a revision integer, or the string
        :func:`fingerprint` returned for a revision-0 file, and a mismatch
        raises :class:`RevisionMismatch` having written nothing.

        Without it the write is exactly what it always was, because fifteen
        callers in this repository and an unknown number outside it rely on
        that -- the guard is opt-in at this level and mandatory one level up.
        """
        path = self.path_for(conversation.character, conversation.identifier)
        if expected is not ChatStore.UNCHECKED:
            current = self.token(conversation.character, conversation.identifier)
            if current != expected:
                raise RevisionMismatch(current)
        conversation.updated = time.time()
        atomic_write_json(path, conversation.to_dict())
        return path

    def token(self, character: str, identifier: str):
        """What this file's revision compares as, right now. ``None`` if absent.

        An integer once anything guarded has written it, the byte fingerprint
        before that. Read and never written: a caller asking what the token is
        must not be the reason a file changes.
        """
        path = self.path_for(character, identifier)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            raise
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return fingerprint(raw)
        if not isinstance(data, dict):
            return fingerprint(raw)
        revision, damaged = _revision_of(data)
        if damaged:
            raise RevisionMismatch(None, "Conversation metadata needs repair")
        return revision if revision else fingerprint(raw)

    def load(self, character: str, identifier: str) -> Conversation:
        path = self.path_for(character, identifier)
        if not path.is_file():
            raise FileNotFoundError(f"No chat {identifier}")
        try:
            raw = path.read_bytes()
        except OSError:
            raw = b""
        conversation = Conversation.from_dict(read_json(path))
        # What this copy compares as, worked out from the bytes that were
        # actually read rather than from a second read a moment later. Two
        # reads would be two different files if something landed between them,
        # which is the whole class of bug the token exists to catch.
        conversation.comparison = (conversation.revision
                                   if conversation.revision else fingerprint(raw))
        return conversation

    def listing(self, character: str) -> list[ChatInfo]:
        """The character's chats, most recently used first."""
        rows = []
        for path in self.folder(character).glob("*.json"):
            try:
                data = read_json(path)
            except (OSError, ValueError):
                continue          # one unreadable chat must not hide the rest
            rows.append(ChatInfo(identifier=str(data.get("id") or path.stem),
                                 title=str(data.get("title") or UNTITLED),
                                 updated=float(data.get("updated", 0.0) or 0.0)))
        return sorted(rows, key=lambda row: row.updated, reverse=True)

    def delete(self, character: str, identifier: str) -> None:
        """Remove one chat, and write down that this id has been used.

        The tombstone is the half that is new, and it is what makes
        :meth:`_identifier`'s promise hold in the other direction. Reserving a
        name stops two pages inventing the same one *at the same time*; a
        tombstone stops a name coming back round after the file at it has gone,
        which is how a reply still in flight when its thread was deleted would
        otherwise be saved into a thread somebody else has since created.
        """
        path = self.path_for(character, identifier)
        final = None
        try:
            final = self.token(character, identifier)
        except (OSError, RevisionMismatch):
            final = None
        path.unlink(missing_ok=True)
        self._entomb(character, identifier, final if isinstance(final, int) else 0)

    # -- tombstones ------------------------------------------------------- #

    TOMBSTONES = ".tombstones.json"
    """Where a character's spent identifiers are written down.

    A dotfile so ``listing()``'s ``*.json`` glob never sees it, beside the
    chats rather than in a database because everything else about this store is
    a file in that folder and a second storage mechanism for one list would be
    one more thing to keep in step.
    """

    MAX_TOMBSTONES = 4096
    """How many spent ids are remembered per character, newest kept.

    Bounded because it is a file that only ever grows, and a chat folder with
    four thousand deletions in it has long since stopped being able to produce
    a timestamp collision with any of them by chance: the ids carry six random
    characters as well as a second.
    """

    def tombstones(self, character: str) -> dict:
        try:
            data = read_json(self.folder(character) / self.TOMBSTONES)
        except (OSError, ValueError):
            return {}
        found = data.get("deleted")
        return found if isinstance(found, dict) else {}

    def entombed(self, character: str, identifier: str) -> bool:
        """Whether this identifier names a chat that has been deleted."""
        return str(identifier) in self.tombstones(character)

    def _entomb(self, character: str, identifier: str, final_revision: int) -> None:
        kept = self.tombstones(character)
        kept[str(identifier)] = {"final_revision": int(final_revision or 0),
                                 "deleted_at": time.time()}
        if len(kept) > self.MAX_TOMBSTONES:
            oldest = sorted(kept.items(), key=lambda row: row[1].get("deleted_at", 0.0))
            for key, _ in oldest[:len(kept) - self.MAX_TOMBSTONES]:
                kept.pop(key, None)
        try:
            atomic_write_json(self.folder(character) / self.TOMBSTONES, {"deleted": kept})
        except OSError:
            # A tombstone that cannot be written is a smaller problem than a
            # deletion that does not happen, and the file is already gone by
            # here. The id simply stays reusable, which is where it was before
            # this existed.
            pass

    def branch(self, conversation: Conversation, index: int) -> Conversation:
        branched = conversation.branch(index, self._identifier(conversation.character))
        self.save(branched)
        return branched

    ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"
    """Base32's digits, lower-cased: a name that is a file name on every host.

    No uppercase, so a case-insensitive filesystem cannot fold two ids into
    one; no ``0``/``1``, so nothing here has to be read back off a screen and
    guessed at.
    """

    def _identifier(self, character: str) -> str:
        """A file name that sorts by time and cannot collide at all.

        It used to be a timestamp with ``-2`` appended if a file was already
        there, which is a check and a create with a gap between them: two pages
        opening a thread in the same second both looked, both saw nothing, and
        both wrote the same file -- the second one over the first one's
        conversation.

        So the name carries six random characters as well as the second, and
        the file at it is *created here*, exclusively, rather than looked for.
        ``O_CREAT|O_EXCL`` is the operating system answering "was I first?" in
        the same breath as asking, which is the only way two processes can both
        ask and only one be told yes. The empty file it leaves is replaced by
        the first save; ``listing()`` already skips a chat it cannot parse, so
        one that is never saved is invisible rather than broken.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        spent = self.tombstones(character)
        for _ in range(64):
            suffix = "".join(secrets.choice(self.ALPHABET) for _ in range(6))
            candidate = f"{stamp}-{suffix}"
            if candidate in spent:
                continue
            path = self.path_for(character, candidate)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            except FileExistsError:
                continue
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    continue
                # A folder that cannot be written to is not a naming problem
                # and must not be answered with sixty-four more attempts.
                raise
            return candidate
        raise OSError("Could not reserve a name for a new conversation")
