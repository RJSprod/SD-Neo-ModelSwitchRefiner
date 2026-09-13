"""The surface another extension imports. MiniMax H3 only, deliberately.

``docs/21-external-llm-api.md`` is the document written for whoever is on the
other end of this; what follows is why it is shaped the way it is.

    import mc_llm_api

    job = mc_llm_api.submit_minimax("a car chase at dusk", first_frame=picture)
    for event in mc_llm_api.subscribe(job).events():
        ...
    print(mc_llm_api.status(job)["prompt"])

An import and not a URL
-----------------------
The caller is a Forge extension, which means it is code in this same Python
process, and the honest transport between two objects in one process is a
function call. An HTTP route would have meant base64-ing a photograph the
caller already holds as a ``PIL.Image``, posting it to ``localhost`` so this
process could decode it back into the object it started as, and inventing a
credential for a caller that shares our address space and could have read it
out of a module anyway.

What the caller *does* often need is HTTP -- for its own browser panel. That is
why :class:`mc_llm_jobs.Feed` yields dicts with ``seq``, ``event`` and ``data``
already in the shape ``text/event-stream`` wants: an extension with a panel
forwards this feed to it in about five lines, on its own routes, under its own
auth, and this module never grows a web server.

As if somebody had gone to LLM Studio
-------------------------------------
The brief was that an external request should be indistinguishable from a
person using the MiniMax panel, and every departure from that would be a
surprise for somebody. So a request here runs the same
:func:`mc_llm_sessions.minimax` generator, over the same llama-server, under
the same workload lock, with the same vendored WanGP instructions, and files
its result in the same Saved prompts history. It emits the same events. It
waits for an image generation on the same card for the same reason.

The three differences are all consequences of there being no screen:

*It is queued.* A person cannot press Enhance twice at once; a caller can, and
five callers can. :mod:`mc_llm_jobs` puts them in a line.

*It carries an id.* Nobody is watching, so the answer has to be collectable
later.

*It may replace the system prompt.* The panel cannot do this and does not need
to -- ``@@`` in the prompt text already does it for a person typing. A caller
generating requests programmatically should not have to build a magic string,
so the override is a parameter. Both defaults stay readable through
:func:`system_prompts`, which is the other half of being able to override one
honestly.

One image, whichever one it is
------------------------------
A caller may hand over a first frame, a last frame and a reference. Exactly one
of them is captioned, because that is what the vendored path does: WanGP runs
one caption pass and hands the enhancer one ``image_caption:`` line, and
inventing a second line here would be inventing a prompt format the H3 models
were not trained on. :func:`_primary` picks the one that fits the variant, and
the job record names both the one used and the ones that were not, so a caller
never has to guess which of its pictures the prompt was written about.

Nothing here is a security boundary. A caller that can import this module is
already running inside the WebUI's process with its privileges; ``origin`` is a
label for the console and the panel's banner, not a claim anyone checks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mc_llm_jobs as jobs

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

API_VERSION = 1
"""The contract's version. Bumped when something already documented changes
meaning; adding a field, an event name or a keyword argument does not bump it.
"""

FIRST_FRAME = "first_frame"
LAST_FRAME = "last_frame"
REFERENCE = "reference"

SLOTS = (FIRST_FRAME, LAST_FRAME, REFERENCE)
"""The picture slots, in the order they are offered and documented."""

PREFERENCE = {
    "fl2va": (FIRST_FRAME, LAST_FRAME, REFERENCE),
    "ref2va": (REFERENCE, FIRST_FRAME, LAST_FRAME),
}
"""Which slot each variant captions when more than one is filled.

FL2VA writes a prompt for a generation that starts from a frame, so the first
frame is what the prompt should be about; Ref2VA writes about reference
material. The fallbacks exist so that a caller who filled only the "wrong" slot
gets a prompt written about the picture it actually sent rather than a
text-only prompt and no explanation.
"""

Rejected = jobs.Rejected
"""Re-exported so a caller catches one name rather than importing two modules."""

Feed = jobs.Feed
"""Re-exported for the same reason: it is what :func:`subscribe` returns, and a
caller annotating a function that takes one should not have to know where it
lives."""


# --------------------------------------------------------------------------- #
# Asking
# --------------------------------------------------------------------------- #


def submit_minimax(prompt: str, *, variant: str = "", first_frame=None, last_frame=None,
                   reference=None, system_prompt: str | None = None, seed=None,
                   origin: str = "", remember: bool = True) -> str:
    """Queue one H3 prompt. Returns the id to track it by.

    ``prompt`` is the only required argument and must not be blank -- the
    enhancer writes *from* something, and a request with nothing to write from
    is a mistake worth reporting at the call site rather than three minutes
    later as an empty answer.

    ``variant`` is ``"fl2va"`` or ``"ref2va"``; anything else, including the
    default empty string, resolves to FL2VA, which is the model WanGP lists
    first. ``first_frame``, ``last_frame`` and ``reference`` each accept a
    ``PIL.Image``, a path, raw image bytes, or a ``data:`` URL. Any combination
    may be supplied and exactly one is captioned -- see :func:`_primary`.

    ``system_prompt`` replaces this variant's vendored instructions for this one
    request. ``@`` and ``@@`` in the prompt text still mean what they mean, so a
    caller can append to its own override.

    ``seed`` defaults to a fresh random one per request, matching the panel and
    matching WanGP's ``prompt_enhancer_randomize_seed``. Pass an integer to
    reproduce a previous run.

    ``remember`` files the finished prompt in LLM Studio's Saved prompts, which
    is what makes an external request look like one somebody made. Turn it off
    for bulk work that would otherwise bury a person's own history.

    Raises :class:`Rejected` -- never a bare ``ValueError`` -- for everything
    the caller can do something about: a blank prompt, an unreadable picture, a
    full queue, a picture sent to a model that cannot see, LLM Studio switched
    off. The last two are the panel's own refusals: the tab does not exist when
    the setting is off, and the panel declines a picture before starting when
    the model has no projector. "As if somebody had gone to LLM Studio" has to
    include the answers that person would have got at the door.
    """
    if not _enabled():
        raise Rejected("LLM Studio is switched off in this WebUI's settings, so nothing "
                       "can write a prompt. Turn it on under Settings → Model Chain.",
                       "disabled")
    from prompt_master.core.models import RANDOM_SEED, draw_seed
    from prompt_master.minimax import enhancer

    text = str(prompt or "").strip()
    if not text:
        raise Rejected("A prompt is required — the enhancer writes from one.",
                       "empty_prompt")

    chosen = enhancer.variant_of(str(variant or "").strip().casefold())
    supplied = {FIRST_FRAME: first_frame, LAST_FRAME: last_frame, REFERENCE: reference}
    filled = tuple(slot for slot in SLOTS if supplied[slot] is not None)
    used = _primary(chosen, filled)
    if used and _sees() is False:
        raise Rejected("The model running has no vision projector, so a picture cannot be "
                       "sent to it. Choose one in LLM Studio → Setup, or send the prompt "
                       "without pictures.", "no_vision")
    image = _data_url(supplied[used], used) if used else None

    resolved = RANDOM_SEED if seed is None else int(seed)
    if resolved == RANDOM_SEED:
        resolved = draw_seed()

    override = None if system_prompt is None else str(system_prompt)
    if override is not None and not override.strip():
        raise Rejected("A system prompt override cannot be blank. Leave it out to use "
                       "the default instructions for this variant.", "empty_system_prompt")

    return jobs.submit(jobs.Job(
        kind="minimax", origin=str(origin or "")[:120], variant=chosen, prompt=text,
        seed=resolved, system=override, images=filled, image_used=used,
        image_ignored=tuple(slot for slot in filled if slot != used),
        image_name=_image_name(supplied[used], used) if used else "",
        remember=bool(remember), _image=image)).identifier


def _enabled() -> bool:
    """Whether the LLM half exists at all, by the same setting that builds the tab."""
    try:
        import mc_llm_studio

        return bool(mc_llm_studio.enabled())
    except Exception:
        logger.debug("Model Chain: could not read whether LLM Studio is enabled",
                     exc_info=True)
        return True


def _sees() -> bool | None:
    """Whether the configured model can take a picture. ``None`` when unreadable.

    ``None`` and not ``False`` on a failure to read, so that a configuration
    this module cannot inspect is left to the run to judge -- which will refuse
    with the vendored client's own sentence if it must -- rather than refused
    here on a guess.
    """
    try:
        import mc_llm_runtime

        return bool(mc_llm_runtime.config().sees)
    except Exception:
        logger.debug("Model Chain: could not read whether the model can see", exc_info=True)
        return None


def _image_name(picture, slot: str) -> str:
    """What the saved history calls the picture: the panel's rule, then the slot.

    A file contributes its basename and never its path -- ``mc_llm_ui
    .picked_name``'s rule, for its reason -- and a picture that arrived with no
    name at all is called by the slot it filled, which is at least a true
    sentence about where it came from.
    """
    import mc_llm_ui as ui

    if isinstance(picture, (str, Path)) and not str(picture).startswith("data:"):
        return ui.picked_name(picture) or slot
    return slot


def _primary(variant: str, filled: tuple) -> str:
    """Which supplied slot gets captioned. Empty when none was supplied."""
    for slot in PREFERENCE.get(variant, SLOTS):
        if slot in filled:
            return slot
    return ""


def _data_url(picture, slot: str) -> str:
    """One caller-supplied picture as the data URL local inference is sent.

    Four input shapes because a caller in this process legitimately has any of
    them, and re-encoding one to a temporary file so it could be read back would
    be two extra copies of somebody's photograph on their disk. A ``data:`` URL
    is passed through untouched: it has already been through the vendored
    preprocessor or it came from somewhere that speaks the same language, and
    re-decoding it here would only be a chance to lose its EXIF orientation.
    """
    import mc_llm_ui as ui

    try:
        if isinstance(picture, (bytes, bytearray, memoryview)):
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(bytes(picture))) as decoded:
                found = ui.data_url(decoded.copy())
        elif isinstance(picture, str) and picture.startswith("data:"):
            found = picture
        elif isinstance(picture, (str, Path)):
            found = ui.data_url(Path(picture))
        else:
            found = ui.data_url(picture)
    except Rejected:
        raise
    except Exception as exc:
        raise Rejected(f"The {slot.replace('_', ' ')} could not be read: {exc}",
                       "bad_image") from exc
    if not found:
        raise Rejected(f"The {slot.replace('_', ' ')} could not be read.", "bad_image")
    return found


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #


def status(job_id: str) -> dict | None:
    """Everything known about one request, or ``None`` once it is forgotten.

    ``None`` and not an exception, because "I no longer have a record of that"
    is an ordinary answer fifteen minutes after a request finished, and a caller
    polling an id it kept in a database should not have to catch something to
    handle it.
    """
    found = jobs.job(job_id)
    return found.describe() if found is not None else None


def result(job_id: str) -> str:
    """Just the written prompt. Empty until there is one."""
    found = jobs.job(job_id)
    return found.result if found is not None else ""


def queue(*, limit: int = 20, origin: str | None = None) -> dict:
    """The queue in one consistent read: running, waiting, recently done.

    ``origin`` narrows the listings to one caller's own requests. ``active``
    and ``waiting`` stay the whole queue's, because a caller's next request is
    behind everything that is there and not only behind its own.
    """
    return jobs.snapshot(limit=limit, origin=origin)


def busy() -> bool:
    """Whether anything external is running or waiting."""
    return jobs.active()


def cancel(job_id: str, reason: str = "") -> dict:
    """Stop one request, queued or running. See :func:`mc_llm_jobs.cancel`."""
    return jobs.cancel(job_id, reason)


def cancel_all(origin: str, reason: str = "") -> dict:
    """Cancel every request one caller made, running or waiting.

    The shape a caller wants when its own panel closes. ``origin`` is required
    here where :func:`mc_llm_jobs.cancel_all` leaves it optional, because "cancel
    everything, whoever asked for it" is the panel's decision to make and not a
    thing a caller should reach by forgetting an argument.
    """
    label = str(origin or "").strip()
    if not label:
        raise Rejected("cancel_all needs the origin you submitted under. To cancel "
                       "every request regardless of origin, use mc_llm_jobs.cancel_all.",
                       "empty_origin")
    return jobs.cancel_all(reason, origin=label)


def subscribe(job_id: str = "", *, ttl: float = jobs.FEED_TTL, cursor: int = 0):
    """A live feed of one request's events, or of the whole queue.

    ``cursor`` resumes: pass the ``seq`` of the last event already handled (or,
    for a queue-wide feed, the last ``stream``) and only what came after it is
    delivered. Zero, the default, replays the request from the moment it was
    queued -- which is what a caller subscribing immediately after submitting
    wants, and which makes the gap between those two calls harmless.
    """
    return jobs.subscribe(job_id, ttl=ttl, cursor=cursor)


def forget(job_id: str) -> bool:
    """Drop a finished request's record now instead of at retention."""
    return jobs.forget(job_id)


# --------------------------------------------------------------------------- #
# Asking what is possible before asking for it
# --------------------------------------------------------------------------- #


def variants() -> tuple:
    """``(("fl2va", "FL2VA — from text or a frame"), ...)``."""
    from prompt_master.minimax import enhancer

    return tuple(enhancer.VARIANTS)


def system_prompt(variant: str = "", *, has_image: bool = False) -> str:
    """The default instructions for one variant, in one of its two forms.

    Two forms because the vendored path has two: WanGP reads a different
    instruction set when the generation has a picture than when it does not, and
    a caller that asked for "the FL2VA system prompt" without saying which would
    be handed one of them arbitrarily.
    """
    from prompt_master.minimax import enhancer

    return enhancer.instructions(enhancer.variant_of(str(variant or "").strip().casefold()),
                                 bool(has_image))


def system_prompts() -> dict:
    """All four defaults, keyed by variant and then by ``"text"``/``"image"``.

    Alongside each, what that variant's prompt is *made of* -- the structure
    guide WanGP shows beside its own prompt box -- because a caller writing an
    override is the one caller that most needs to know what the default was
    trying to produce.
    """
    from prompt_master.minimax import enhancer

    return {variant: {"label": label,
                      "text": enhancer.instructions(variant, False),
                      "image": enhancer.instructions(variant, True),
                      "structure": enhancer.infos(variant),
                      "max_tokens": enhancer.max_tokens(variant)}
            for variant, label in enhancer.VARIANTS}


def capabilities() -> dict:
    """What this installation can do right now, so a caller can check first.

    Every field is a fact about *this* machine at *this* moment, not a promise:
    ``vision`` can go false when somebody switches to a model with no projector,
    and a request submitted a second later will fail with that reason rather
    than silently drop its picture. Reading this before submitting turns that
    into a message the caller can show instead of a job that failed.
    """
    found = {
        "api_version": API_VERSION,
        "kinds": ["minimax"],
        "enabled": _enabled(),
        "variants": [value for value, _ in variants()],
        "slots": list(SLOTS),
        "events": list(jobs.EVENTS),
        "max_queued": jobs.MAX_QUEUED,
        "feed_ttl": jobs.FEED_TTL,
        "job_retention": jobs.JOB_RETENTION,
        "configured": False,
        "vision": False,
        "model": "",
        "reason": "",
    }
    try:
        import mc_llm_runtime

        configuration = mc_llm_runtime.config()
        found["configured"] = bool(configuration.configured)
        found["vision"] = bool(configuration.sees)
        found["model"] = Path(str(configuration.model or "")).name
        if not found["enabled"]:
            found["reason"] = ("LLM Studio is switched off in this WebUI's settings. "
                               "Requests will be refused until it is turned on.")
        elif not found["configured"]:
            found["reason"] = ("No language model is set up. Choose one in LLM Studio → "
                               "Setup.")
        elif not found["vision"]:
            found["reason"] = ("The model running has no vision projector, so requests "
                               "carrying a picture will fail. Text-only requests are fine.")
    except Exception as exc:
        logger.debug("Model Chain: could not read the LLM configuration for the "
                     "external API", exc_info=True)
        found["reason"] = f"The language-model configuration could not be read: {exc}"
    return found
