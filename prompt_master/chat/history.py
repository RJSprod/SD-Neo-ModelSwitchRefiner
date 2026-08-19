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
    # A still, as the data URL the vision request carries. Stored rather than
    # referenced: the file it came from can be moved or deleted, and a
    # conversation that then cannot be replayed is a conversation that lied
    # about what it sent.
    image: str = ""
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

    def to_dict(self) -> dict:
        return {"role": self.role, "versions": list(self.versions), "active": self.active,
                "image": self.image, "image_name": self.image_name}

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
                   image=str(data.get("image", "")), image_name=str(data.get("image_name", "")))


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

    # ── editing ──────────────────────────────────────────────────────────────

    def append(self, role: str, text: str = "", image: str = "", image_name: str = "") -> Message:
        message = Message(role=role, versions=[text], image=image, image_name=image_name)
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
        return {"id": self.identifier, "character": self.character, "title": self.title,
                "created": self.created, "updated": self.updated,
                "response_prefix": self.response_prefix,
                "messages": [message.to_dict() for message in self.messages]}

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        messages = data.get("messages")
        return cls(
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

    def save(self, conversation: Conversation) -> Path:
        conversation.updated = time.time()
        path = self.path_for(conversation.character, conversation.identifier)
        atomic_write_json(path, conversation.to_dict())
        return path

    def load(self, character: str, identifier: str) -> Conversation:
        path = self.path_for(character, identifier)
        if not path.is_file():
            raise FileNotFoundError(f"No chat {identifier}")
        return Conversation.from_dict(read_json(path))

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
        self.path_for(character, identifier).unlink(missing_ok=True)

    def branch(self, conversation: Conversation, index: int) -> Conversation:
        branched = conversation.branch(index, self._identifier(conversation.character))
        self.save(branched)
        return branched

    def _identifier(self, character: str) -> str:
        """A file name that sorts by time and cannot collide within a second."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate, index = stamp, 2
        while self.path_for(character, candidate).exists():
            candidate, index = f"{stamp}-{index}", index + 1
        return candidate
