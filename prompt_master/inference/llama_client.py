from __future__ import annotations

from collections.abc import Callable
import threading
import httpx

from prompt_master.models.managed_profiles import SAMPLER_FIELDS

from .local_only import check_base_url, check_messages
from .sse import assistant_chunks


REQUEST_SAMPLER_FIELDS = ("min_p", "top_k", "dynatemp_range", "xtc_probability",
                          "xtc_threshold", "repeat_penalty", "presence_penalty",
                          "frequency_penalty", "repeat_last_n", "dry_multiplier", "dry_base",
                          "dry_allowed_length", "dry_penalty_last_n")
"""The sampler keys one request may carry beyond temperature and top_p.

A conversation thread's own choices (``mc_llm_generation`` in the extension
around this package), applied after the profile's fixed fields so a thread's
choice wins. A whitelist for the profile's reason: whatever reaches here ends up
in a JSON body, and "whatever the caller passed" is not a thing to put in one.
Every name is llama-server's own (``tools/server/server-schema.cpp``).
"""


def request_sampling(sampling: dict | None) -> dict:
    """``sampling`` filtered to :data:`REQUEST_SAMPLER_FIELDS`, numbers only."""
    if not sampling:
        return {}
    return {key: sampling[key] for key in REQUEST_SAMPLER_FIELDS
            if key in sampling and isinstance(sampling[key], (int, float))
            and not isinstance(sampling[key], bool)}


def request_bias(logit_bias) -> dict:
    """``{"logit_bias": [...]}`` for a request, or ``{}``.

    llama-server's own field: ``[token, strength]`` pairs, the token an id or a
    text the server reads itself, the strength a number or ``False`` for never.
    Anything else in the list is left out, for the reason the sampler fields are
    a whitelist. A thread's injected words (``mc_llm_injection`` in the
    extension around this package) are the only caller.
    """
    found = []
    for pair in logit_bias or ():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        token, strength = pair
        if isinstance(token, bool) or not isinstance(token, (int, str)):
            continue
        if isinstance(token, str) and not token:
            continue
        if strength is False or (isinstance(strength, (int, float))
                                 and not isinstance(strength, bool)
                                 and strength == strength and abs(strength) < float("inf")):
            found.append([token, strength])
    return {"logit_bias": found} if found else {}


class LlamaClient:
    """A client for the local llama-server, and for nothing else.

    The endpoint is llama.cpp's OpenAI-compatible chat API, which describes the
    shape of the JSON below and no more than that: every request this class
    makes goes to a llama-server this application started on the loopback
    interface. It is checked rather than documented -- see
    :mod:`prompt_master.inference.local_only` -- because a base URL is a string,
    and the constructor is the last place a prompt has not yet been attached to
    one. There is no hosted fallback in this class or beneath it: a local
    failure is raised locally.
    """

    def __init__(self, base_url: str, api_key: str, sampling: dict | None = None):
        # sampling is the managed backbone profile's fixed sampler fields, or
        # nothing at all on a manual install. It is filtered here rather than
        # trusted, and filtered against the same whitelist the profiles are
        # written against: everything in it ends up in a JSON body, and
        # "whatever the caller passed" is not a thing to put in one. temperature
        # and top_p are deliberately not on that list — they arrive per request
        # from Creative Mode's 0-10 curve, and a profile that set them would be
        # overriding the user's own Creativity setting from a checked-in file.
        self.base_url, self.api_key = check_base_url(base_url), api_key
        self.sampling = {key: sampling[key] for key in SAMPLER_FIELDS
                         if sampling and key in sampling and sampling[key] is not None}

    def stream_chat(self, messages: list[dict], max_tokens: int, seed: int, on_text: Callable[[str], None], cancel: threading.Event | None = None, temperature: float = 0.85, top_p: float = 0.95, sampling: dict | None = None, logit_bias: list | None = None) -> str:
        # Before the body is built, so an image llama-server would have had to
        # fetch never reaches a request at all. See local_only.check_messages.
        check_messages(messages)
        # Writer defaults match upstream backend.chat_stream (0.85 / 0.95); the
        # smart-negative pass overrides them with upstream's cooler 0.3 / 0.9.
        payload = {"model":"prompt-master","messages":messages,"temperature":temperature,"top_p":top_p,"max_tokens":max_tokens,"seed":seed,"stream":True,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":False,"thinking":False}}
        payload.update(self.sampling)
        # A request's own sampler fields after the profile's: a conversation
        # thread's choice of top-k beats the catalogue's (REQUEST_SAMPLER_FIELDS).
        payload.update(request_sampling(sampling))
        payload.update(request_bias(logit_bias))
        pieces = []
        with httpx.Client(timeout=httpx.Timeout(600, connect=10), headers={"Authorization":f"Bearer {self.api_key}"}) as client:
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for chunk in assistant_chunks(response.iter_lines()):
                    if cancel and cancel.is_set(): return "".join(pieces)
                    pieces.append(chunk); on_text(chunk)
        return "".join(pieces)
