"""The two guards that keep local inference local.

llama.cpp exposes an OpenAI-compatible chat schema. Those words describe the
shape of a JSON body and nothing else: in this project that schema is spoken
only to a llama-server this process started, listening on the loopback
interface, and the phrase "OpenAI-compatible" authorises no communication with
api.openai.com or with any other hosted model service (design intent section
15). There is no cloud fallback anywhere in this package, and these two
functions are what makes that a property of the code rather than a promise in a
docstring.

*Loopback only.* A base URL is a string, and a string is a thing that can come
from a settings box, a JSON file or an environment variable. Every one of those
is a way for prompts, character cards, Creative briefs and Spatial data to
leave the machine, so the address is checked once, at construction, before any
of them can be attached to it (invariant I-9).

*Embedded images only.* llama.cpp will happily fetch an ``image_url`` whose URL
is remote, which would make the *server* perform a data-dependent network
request on a user's behalf. The image content this application sends is always
a ``data:`` URI it built itself; anything else is refused here rather than
handed downstream, so a remote URL that reached a message list by any route
fails locally instead of quietly turning into a fetch (invariant I-11).
"""

from __future__ import annotations

from urllib.parse import urlsplit

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
"""Every address this application will speak inference to.

The whole IPv4 loopback block would be defensible and is deliberately not here:
the runtime binds ``127.0.0.1`` explicitly, so anything else in ``127/8`` is
already not a server this process started, and a guard that admits addresses
the product never produces is a guard with a gap in it for no benefit.
"""

LOCAL_SCHEMES = frozenset({"http", "https"})
"""``http`` in practice. ``https`` is accepted for a loopback host so that a
user who puts a local TLS terminator in front of their own llama-server is not
refused for doing something safer than what was asked of them."""

EMBEDDED_PREFIX = "data:"
"""What an image sent for inference has to be. See :func:`check_messages`."""


class NotLocal(ValueError):
    """An inference destination or payload that would leave this machine.

    A ``ValueError`` because that is what a bad argument is, and callers that
    already report configuration problems as sentences need no new branch to
    say this one.
    """


def check_base_url(base_url: str) -> str:
    """``base_url`` if it addresses a local llama-server, else raise.

    Returned rather than merely validated so a caller can write
    ``self.base_url = check_base_url(url)`` and have no way to keep the
    unchecked one by accident.
    """
    text = str(base_url or "").strip().rstrip("/")
    if not text:
        raise NotLocal("An inference endpoint is required, and it must be a local "
                       "llama-server on 127.0.0.1.")
    parts = urlsplit(text)
    if parts.scheme.casefold() not in LOCAL_SCHEMES or not parts.netloc:
        raise NotLocal(
            f"{text} is not a usable local inference endpoint. Inference in this extension "
            f"runs against a llama-server this process started on 127.0.0.1."
        )
    if parts.username or parts.password:
        raise NotLocal("An inference endpoint must not carry credentials; the local "
                       "llama-server is addressed by host and port alone.")
    host = (parts.hostname or "").casefold()
    if host not in LOOPBACK_HOSTS:
        raise NotLocal(
            f"Refusing to send inference to {parts.netloc}. Prompts, images and replies stay "
            f"on this machine: the only inference endpoints this extension will use are the "
            f"local llama-server it starts on 127.0.0.1. There is no cloud fallback."
        )
    return text


def check_messages(messages) -> None:
    """Refuse a chat payload carrying an image llama-server would have to fetch.

    Checked at the point of transmission rather than at the point every image is
    built. There are five callers that build image content -- Conversation,
    Prompt Studio's i2v pass, MiniMax, Krea's reference captioner and the prompt
    engine -- and all five already produce ``data:`` URIs; a guard placed in
    each of them would be five things to remember the next time a sixth is
    written, and this is one.
    """
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            _check_image_url(url)


def _check_image_url(url) -> None:
    text = str(url or "").strip()
    if text.casefold().startswith(EMBEDDED_PREFIX):
        return
    shown = text[:120] + "…" if len(text) > 120 else text
    raise NotLocal(
        f"Refusing to send an image by reference ({shown or 'an empty image URL'}). Images for "
        f"local inference are embedded as data:image/…;base64 content, so llama-server never "
        f"fetches anything on your behalf."
    )


__all__ = ["EMBEDDED_PREFIX", "LOCAL_SCHEMES", "LOOPBACK_HOSTS", "NotLocal",
           "check_base_url", "check_messages"]
