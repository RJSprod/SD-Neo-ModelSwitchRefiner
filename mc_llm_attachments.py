"""Where a conversation's pictures live, and how a transcript shows them.

A chat used to carry its attachments inside itself: the JPEG the model was
shown, base64-encoded, in the same JSON file as the words. That is durable and
it is honest -- what the file holds is exactly what was sent -- and it is wrong
in three ways that only appear once somebody actually uses it.

*It cannot be shown.* The transcript is streamed to the browser on every token,
so a picture inside a message is re-sent on every token: a hundred and fifty
kilobytes per chunk, per image, for the length of a reply. What that buys is a
line of italic text saying a picture was attached, because embedding the thing
itself was never affordable.

*It cannot be found.* A picture inside a JSON file is not a picture on a disk.
There is no folder to open, nothing to look through, and no way to delete one
without editing a conversation by hand.

*It cannot be shared between turns.* The same picture attached twice is stored
twice, and a thread branched five times is six copies.

So the bytes go on disk, under a folder this extension owns and a person can
open, and the conversation keeps a short relative path to them. The transcript
then shows the picture itself through the host's own ``file=`` route, which
costs a path per token instead of a photograph.

    <LLM data root>/chat-images/<character>/<content hash>.jpg

Content-addressed inside the character's folder: the same picture attached
twice is one file, a branch shares its parent's pictures rather than copying
them, and an edit that re-attaches the same photograph writes nothing. Filed by
character so the folder can be browsed by somebody looking for "the pictures I
sent to this character", which is the way anybody would go looking.

Nothing here deletes. A picture may be reachable from several threads and from
several branches of one thread, so no thread can know whether it is the last
one holding it -- and a "tidy up" that guessed wrong would take away part of a
conversation. The folder is the user's; the request was to be able to find and
delete these by hand, and that is exactly what it is for.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.parse import quote

import mc_llm_paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

DIRNAME = "chat-images"
"""The folder, under the LLM data root, beside ``chats`` and ``characters``."""

SUFFIX = ".jpg"
"""What is written. One format because there is one encoder: every picture that
reaches here has already been through the vendored preprocessor, which produces
the sized JPEG that inference is sent."""

NAME_LENGTH = 32
"""How much of the SHA-256 names the file. Long enough that a collision is not
a thing to reason about, short enough that the folder is readable."""

MISSING = "*[the picture for this message is no longer on disk]*"
"""What a transcript says where a picture used to be.

Said rather than skipped. A message that was sent with a picture is not the
same message without one, and a transcript that quietly drops it is a
transcript that misrepresents the conversation -- particularly this one, where
the reply underneath is about a picture the reader can no longer see.
"""


def root() -> Path:
    """The folder every conversation's pictures live under. Creates nothing."""
    return mc_llm_paths.data_root() / DIRNAME


def folder(character: str) -> Path:
    """Where ``character``'s pictures live.

    Through the chat store's own name-safety rule rather than a second one:
    these folders sit beside the ones the chats are filed in, and two different
    answers to "what is this character's folder called" is how a rename comes
    to lose half of itself.
    """
    from prompt_master.chat.characters import safe_stem

    return root() / safe_stem(character or "unnamed")


def store(picture, character: str) -> str:
    """Write a picked picture into ``character``'s folder. Returns the record.

    The record is relative to :func:`root`, so an installation that is moved to
    another drive keeps its conversations whole -- the same reason the state
    file records a model relatively.

    Idempotent by construction: the name is the hash of the bytes, so storing
    the same picture again finds the file already there and writes nothing.
    """
    from prompt_master.imaging.preprocess import jpeg_bytes

    raw = picture if isinstance(picture, (bytes, bytearray)) else jpeg_bytes(picture)
    return _write(bytes(raw), character)


def _write(raw: bytes, character: str) -> str:
    digest = hashlib.sha256(raw).hexdigest()[:NAME_LENGTH]
    destination = folder(character) / f"{digest}{SUFFIX}"
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and renamed, so a crash mid-write cannot leave a
        # truncated file under a name that says what its contents hash to.
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(raw)
        partial.replace(destination)
        logger.debug("Model Chain: kept a conversation attachment at %s", destination)
    return _record(destination)


def _record(path: Path) -> str:
    """``<character>/<hash>.jpg``, or the absolute path if it escaped the folder."""
    try:
        return path.resolve().relative_to(root().resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def locate(recorded: str) -> Path | None:
    """The file a record names, or ``None`` when it is not there any more.

    Contained rather than merely joined. A record comes out of a JSON file, and
    a JSON file is a thing somebody can edit: ``../../`` in it must resolve to
    nothing rather than to somewhere else on the disk.
    """
    if not recorded:
        return None
    try:
        base = root().resolve()
        candidate = Path(recorded)
        found = (candidate if candidate.is_absolute() else base / candidate).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_absolute() and base not in found.parents:
        return None
    return found if found.is_file() else None


def data_url(recorded: str) -> str:
    """A stored picture as the embedded URL a local inference request carries.

    Read back at the moment the request is built rather than held in the
    conversation, which is the whole point of the folder: the words live in the
    chat file and the pictures live beside it, and only what is actually being
    sent is ever encoded.
    """
    found = locate(recorded)
    if found is None:
        return ""
    try:
        from prompt_master.imaging.preprocess import as_data_url

        return as_data_url(found.read_bytes())
    except OSError:
        logger.warning("Model Chain: could not read the attachment at %s", found, exc_info=True)
        return ""


def markup(recorded: str, alt: str = "") -> str:
    """One stored picture, as the transcript should show it.

    An ``<img>`` through the host's own ``file=`` route and never a ``data:``
    URL, and the reason is the streaming: the whole transcript is re-sent to
    the browser on every token of a reply, so a picture embedded in it is
    re-sent on every token too. A path is a hundred bytes and a photograph is a
    hundred and fifty thousand.

    The URL is relative so that it resolves against wherever the application is
    served from, which is not always the root of the host.
    """
    found = locate(recorded)
    if found is None:
        return MISSING
    allow(found)
    source = quote(found.as_posix(), safe="/:")
    label = _escape(alt or "attached image")
    return f'<img src="file={source}" alt="{label}" class="mc-llm-attached">'


_allowed: set = set()
"""Which files the host has already been asked to serve.

Not an optimisation for its own sake. :func:`markup` runs once per message
every time the transcript is rendered, and the transcript is rendered on every
token of a reply -- so without this, a thread with three pictures in it asks
the host to allow three files twenty times a second, and the host's own answer
to that question rebuilds a set each time.
"""


def allow(path: Path) -> None:
    """Ask the host to serve this file. Best-effort, and never raises.

    Gradio will not serve a path it was not told about. The WebUI's own gallery
    has the same problem and solves it with ``modules.ui_tempdir``, so that is
    what is called rather than a second mechanism: it is the host's answer to
    this question, it is what the host's own images go through, and it keeps
    this extension out of Gradio's internals.

    Most installations need none of it -- the WebUI launches with its data
    directory already on Gradio's allow-list, and the LLM data root is inside
    it -- so a failure here is usually a picture that was going to be served
    anyway. It is never a reason to fail rendering a transcript.
    """
    named = str(path)
    if named in _allowed:
        return
    try:
        from modules import shared, ui_tempdir

        ui_tempdir.register_tmp_file(shared.demo, named)
        _allowed.add(named)
    except Exception:
        logger.debug("Model Chain: could not register %s with the host's file route", path,
                     exc_info=True)


def adopt(conversation, character: str = "") -> bool:
    """Move a conversation's inline pictures onto disk. Returns whether any moved.

    The migration for every chat written before there was a folder to put them
    in. It runs when such a chat is opened, once, and what it produces is a
    conversation that is smaller, that shows its pictures, and whose
    attachments are files somebody can look at.

    Total: a picture that cannot be decoded or cannot be written is left
    exactly where it is, still inside the message, still sent to the model.
    Losing an attachment to a tidying-up would be a far worse outcome than a
    chat that keeps carrying one.
    """
    import base64

    moved = False
    who = character or getattr(conversation, "character", "") or ""
    for message in getattr(conversation, "messages", None) or ():
        inline = getattr(message, "image", "")
        if not inline or getattr(message, "image_path", ""):
            continue
        _, _, encoded = str(inline).partition(",")
        try:
            message.image_path = _write(base64.b64decode(encoded, validate=True), who)
        except Exception:
            logger.warning("Model Chain: could not move an attachment out of a conversation "
                           "and onto disk; it stays inside the chat file", exc_info=True)
            continue
        message.image = ""
        moved = True
    return moved


def _escape(text: str) -> str:
    import html

    return html.escape(str(text or ""), quote=True)


__all__ = ["DIRNAME", "MISSING", "adopt", "allow", "data_url", "folder", "locate",
           "markup", "root", "store"]
