from __future__ import annotations

from collections.abc import Callable
import threading
import httpx

from prompt_master.models.managed_profiles import SAMPLER_FIELDS

from .local_only import check_base_url, check_messages
from .sse import assistant_chunks


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

    def stream_chat(self, messages: list[dict], max_tokens: int, seed: int, on_text: Callable[[str], None], cancel: threading.Event | None = None, temperature: float = 0.85, top_p: float = 0.95) -> str:
        # Before the body is built, so an image llama-server would have had to
        # fetch never reaches a request at all. See local_only.check_messages.
        check_messages(messages)
        # Writer defaults match upstream backend.chat_stream (0.85 / 0.95); the
        # smart-negative pass overrides them with upstream's cooler 0.3 / 0.9.
        payload = {"model":"prompt-master","messages":messages,"temperature":temperature,"top_p":top_p,"max_tokens":max_tokens,"seed":seed,"stream":True,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":False,"thinking":False}}
        payload.update(self.sampling)
        pieces = []
        with httpx.Client(timeout=httpx.Timeout(600, connect=10), headers={"Authorization":f"Bearer {self.api_key}"}) as client:
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for chunk in assistant_chunks(response.iter_lines()):
                    if cancel and cancel.is_set(): return "".join(pieces)
                    pieces.append(chunk); on_text(chunk)
        return "".join(pieces)
