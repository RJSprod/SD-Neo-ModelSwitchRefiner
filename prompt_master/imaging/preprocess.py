from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}

MAX_SIDE = 768
"""What a picture is shrunk to before it is sent to the local model.

A vision projector charges by the tile, and an 8-megapixel phone photograph is
several times the context of the conversation it is attached to.
"""


def image_data_url(path: Path) -> str:
    """A picture on disk, as the embedded data URL local inference is sent."""
    if path.suffix.casefold() not in SUPPORTED:
        raise ValueError("Only PNG, JPEG, and WebP images are supported")
    with Image.open(path) as source:
        return encode(source)


def jpeg_bytes(image: Image.Image) -> bytes:
    """One already-decoded picture, sized and encoded the way inference wants it.

    The bytes rather than a data URL, because they are also what gets written
    to disk when a conversation keeps its attachments: the picture a chat shows
    a week later is then byte-for-byte the picture the model was shown, and no
    second encoding can drift from the first.

    The transpose and the RGB conversion stay even where the caller has already
    done them -- both are idempotent, and this is the function that has to be
    right rather than the four call sites in front of it.
    """
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    prepared.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    prepared.save(buffer, "JPEG", quality=82, optimize=True)
    return buffer.getvalue()


def as_data_url(raw: bytes) -> str:
    """JPEG bytes as the embedded URL a local inference request carries."""
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def encode(image: Image.Image) -> str:
    """One already-decoded picture, as ``data:image/jpeg;base64,…``.

    Split out from :func:`image_data_url` because the picture does not always
    arrive as a file. Gradio's image component hands back a decoded PIL image
    when it is asked for one, and re-encoding that to a temporary file only to
    read it back would be two extra copies of somebody's photograph on their
    disk for no gain.

    A data URL and never a remote one. llama.cpp will fetch an ``image_url``
    whose URL is remote, which would make the inference server perform a
    network request on the user's behalf; every picture this application sends
    is embedded bytes it produced itself.
    """
    return as_data_url(jpeg_bytes(image))
