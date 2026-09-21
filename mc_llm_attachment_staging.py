"""An uploaded picture, held until somebody presses Send. With a readiness state.

Two composers can now attach a picture and one of them lives in a floating
panel that is not a Gradio component at all. What both need, and what neither
had, is a *name for the picture before the message exists* — so that Send can
refuse while an upload is still going, so that a second picture can visibly
replace the first, and so that a picture chosen in the flyout is the same
picture the tab's chip shows.

The pipeline underneath is untouched. Once a staged file is used, it goes
through ``mc_llm_attachments.store`` and the same vision checks as before; this
module sits in front of that and answers one question — *is this picture ready
to be sent* — with something better than a timer.

A timer is what the alternative always degenerates into, and specification S6
names it: waiting 300 ms after a file input changes and calling that "uploaded"
is a guess that is right on a laptop and wrong on a phone with a 12-megapixel
camera. Readiness here is a state that the decode either reached or did not.

What is refused, and why here
-----------------------------
Decoding is the attack surface. A 300-byte PNG can decode to a 40-gigapixel
canvas, an SVG can carry script, and a "JPEG" can be an HTML document a browser
would happily render from our own origin. So the bytes are sniffed, the
declared dimensions are checked *before* the pixels are read, animation is
refused rather than flattened to a frame nobody chose, and what is kept is a
freshly-encoded raster rather than the file that arrived. Nothing that reaches
``mc_llm_attachments`` from here is a file somebody else wrote.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

MAX_ENCODED_BYTES = 20 * 1024 * 1024
"""The largest upload accepted, as it arrives. Twenty mebibytes.

Comfortably above a phone photograph and far below a video somebody dragged in
by mistake. Checked before anything is decoded, because the cheapest refusal is
the one that never allocates.
"""

MAX_PIXELS = 40_000_000
"""The largest picture accepted, decoded. Forty megapixels.

The decompression-bomb bound. A PNG declares its dimensions in the first
twenty-four bytes, so this is answered before the pixel data is touched: a file
claiming 60,000 x 60,000 is refused having cost a read of its header.
"""

STAGING_TTL = 86400.0
"""How long an unused staged picture is kept. Twenty-four hours.

Long enough that a draft written on a phone in the morning still has its
picture in the evening; short enough that a folder of abandoned uploads is a
day's worth rather than a year's.
"""

MAX_PER_SCOPE = 24
"""How many staged pictures one access scope may hold at once.

A bound on a thing a browser can create, which is the only kind of bound worth
writing down. Oldest unpinned first when it is reached.
"""

UPLOADING = "uploading"
PROCESSING = "processing"
READY = "ready"
FAILED = "failed"
REMOVED = "removed"

STATES = (UPLOADING, PROCESSING, READY, FAILED, REMOVED)

# -- what a supported picture looks like, in its first bytes ---------------- #

SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "PNG"),
    (b"\xff\xd8\xff", "image/jpeg", "JPEG"),
    (b"RIFF", "image/webp", "WEBP"),
)
"""Sniffed rather than trusted. A declared content type is what the *uploader*
says the file is, which is not evidence about the file."""


@dataclass
class Staged:
    """One picture, waiting for a message to be attached to."""

    token: str
    name: str = ""
    state: str = UPLOADING
    reason: str = ""
    scope: str = ""
    width: int = 0
    height: int = 0
    kind: str = ""
    image: object = None
    created: float = field(default_factory=time.time)
    pinned_by: str = ""

    @property
    def ready(self) -> bool:
        return self.state == READY and self.image is not None

    def describe(self) -> dict:
        found = {"token": self.token, "name": self.name, "state": self.state,
                 "width": self.width, "height": self.height, "kind": self.kind,
                 "created": self.created}
        if self.reason:
            found["reason"] = self.reason
        return found


_guard = threading.Lock()
_staged: dict[str, Staged] = {}


class Rejected(Exception):
    """A file that will not be staged, with the sentence to show for it."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# Staging
# --------------------------------------------------------------------------- #


def stage(raw: bytes, name: str = "", scope: str = "local") -> Staged:
    """Validate, decode and keep one uploaded picture. Returns its record.

    The record is returned in whatever state it reached -- ``ready`` or
    ``failed`` -- rather than raising, because both are answers the composer
    shows and only one of them is exceptional to a *caller*. What raises is a
    request that is not a picture at all.
    """
    found = Staged(token=secrets.token_urlsafe(18), name=_clean(name), scope=str(scope or ""),
                   state=PROCESSING)
    try:
        _check(raw)
        image, kind = _decode(raw)
    except Rejected as refusal:
        found.state, found.reason = FAILED, refusal.reason
    except Exception as exc:
        logger.debug("Model Chain: could not stage an attachment", exc_info=True)
        found.state, found.reason = FAILED, f"That picture could not be read: {exc}"
    else:
        found.image, found.kind = image, kind
        found.width, found.height = image.size
        found.state = READY
    with _guard:
        _expire_locked()
        _bound_locked(found.scope)
        _staged[found.token] = found
    return found


def _clean(name: str) -> str:
    """A file name as a label: no path, no control characters, bounded.

    It is shown beside the picture and read by a screen reader, and it came
    from somebody's filesystem. Everything that makes it a *path* is taken off
    here so that nothing downstream has to remember to.
    """
    found = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    found = "".join(character for character in found if character.isprintable())
    return found.strip()[:120]


def _check(raw: bytes) -> None:
    if not raw:
        raise Rejected("That file was empty.")
    if len(raw) > MAX_ENCODED_BYTES:
        raise Rejected("That picture is larger than 20 MB. Choose a smaller one.")
    head = raw[:16]
    if head.lstrip()[:1] in (b"<",):
        # An SVG or an HTML document. Refused by shape rather than by
        # extension: the danger is that it would be *rendered*, and a file that
        # starts with a tag is a file a browser would render.
        raise Rejected("Vector and HTML files cannot be attached. Choose a PNG, JPEG "
                       "or WebP.")
    for signature, _, _ in SIGNATURES:
        if head.startswith(signature):
            return
    raise Rejected("That file is not a PNG, JPEG or WebP picture.")


def _decode(raw: bytes):
    """The bytes as a picture this application owns. Re-encoded, never passed on.

    Three separate refusals before a pixel is read, then a copy. The copy is
    what breaks the link to the uploaded file: what
    ``mc_llm_attachments.store`` receives is an image object this process built,
    so nothing of the original container -- its metadata, its trailing bytes,
    its second frame -- travels any further.
    """
    import io

    from PIL import Image, ImageOps, ImageSequence

    stream = io.BytesIO(raw)
    try:
        image = Image.open(stream)
    except Exception:
        raise Rejected("That picture could not be read.")
    kind = str(getattr(image, "format", "") or "")
    if kind.upper() not in ("PNG", "JPEG", "WEBP", "MPO"):
        raise Rejected("That file is not a PNG, JPEG or WebP picture.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise Rejected("That picture has no size.")
    if width * height > MAX_PIXELS:
        raise Rejected("That picture is too large to work with. "
                       "Forty megapixels is the limit.")
    try:
        frames = sum(1 for _ in ImageSequence.Iterator(image))
    except Exception:
        frames = 1
    if frames > 1:
        raise Rejected("Animated pictures cannot be attached. Choose a still.")
    image.seek(0)
    try:
        # Orientation first, so a phone photograph is the way up it was taken
        # rather than the way up its bytes are. ``exif_transpose`` also drops
        # the tag it honoured, which is what stops a second reader rotating it
        # again.
        upright = ImageOps.exif_transpose(image)
    except Exception:
        upright = image
    return upright.convert("RGB"), kind.upper()


# --------------------------------------------------------------------------- #
# Using and forgetting
# --------------------------------------------------------------------------- #


def resolve(token: str) -> Staged | None:
    """The staged picture a command names, or ``None`` if it is not there."""
    with _guard:
        _expire_locked()
        found = _staged.get(str(token or ""))
        return found if found is not None and found.state != REMOVED else None


def describe(token: str) -> dict | None:
    found = resolve(token)
    return found.describe() if found is not None else None


def pin(token: str, operation_id: str) -> bool:
    """Hold this picture against expiry while an operation references it.

    The expiry sweep would otherwise be able to delete the attachment of a
    message that is halfway through being sent -- which is the one moment it is
    guaranteed to still be needed.
    """
    with _guard:
        found = _staged.get(str(token or ""))
        if found is None:
            return False
        found.pinned_by = str(operation_id or "")
        return True


def consume(token: str) -> None:
    """Mark a staged picture used. Its bytes are no longer needed.

    Not deleted synchronously by a retry-safe caller: the record stays so that
    a repeated command with the same operation id finds the token it named
    rather than "expired", and the image itself is released because the message
    now owns a copy.
    """
    with _guard:
        found = _staged.get(str(token or ""))
        if found is None:
            return
        found.image = None
        found.state = REMOVED
        found.reason = "used"


def discard(token: str) -> bool:
    """Remove a staged picture because somebody took it off the message."""
    with _guard:
        return _staged.pop(str(token or ""), None) is not None


def _expire_locked() -> None:
    now = time.time()
    for token in [token for token, found in _staged.items()
                  if not found.pinned_by and now - found.created > STAGING_TTL]:
        _staged.pop(token, None)


def _bound_locked(scope: str) -> None:
    mine = [(token, found) for token, found in _staged.items()
            if found.scope == scope and not found.pinned_by]
    if len(mine) <= MAX_PER_SCOPE:
        return
    for token, _ in sorted(mine, key=lambda row: row[1].created)[:len(mine) - MAX_PER_SCOPE]:
        _staged.pop(token, None)


def expire() -> int:
    """Drop staged pictures nobody claimed. Returns how many went."""
    with _guard:
        before = len(_staged)
        _expire_locked()
        return before - len(_staged)


def reset() -> None:
    """Drop everything staged. For the tests, and for a UI reload."""
    with _guard:
        _staged.clear()


def pending() -> list:
    with _guard:
        return [found.describe() for found in _staged.values()]
