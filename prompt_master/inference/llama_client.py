from __future__ import annotations

from collections.abc import Callable, Mapping
import threading
import httpx

from .sse import assistant_chunks

SAMPLING_FIELDS: dict[str, type] = {
    "top_k": int,
    "min_p": float,
    "typical_p": float,
    "dynatemp_range": float,
    "dynatemp_exponent": float,
    "xtc_probability": float,
    "xtc_threshold": float,
    "repeat_penalty": float,
    "presence_penalty": float,
    "frequency_penalty": float,
}
"""Optional llama.cpp sampler fields this client will put in a request.

A whitelist and not a passthrough, and the difference matters at both ends. On
the way out, a caller cannot reach into the payload: whatever a panel, a
preferences file or a pasted infotext ends up holding, only these ten names
travel, coerced to the type llama.cpp's server expects, so a UI string can
never arrive at the server as a string where a float was meant. On the way in,
a field llama.cpp does not know about is a field that is simply not here --
which is what "unsupported fields are not sent" means in practice, because a
server that ignores an unknown key silently and one that rejects the whole
request are both outcomes nobody wants to debug from a prompt box.

The list is longer than anything currently sends. ``prompt_master.krea
.variation`` uses two of these today; the rest are named so that adding a row
to that table is the only edit needed when there is a reason to.
"""


def sampling_fields(extra: Mapping | None) -> dict:
    """``extra`` reduced to the whitelisted fields, typed as llama.cpp wants them.

    Unknown names are dropped rather than raising: the caller that supplied one
    is asking for a sampler this runtime has no field for, and the right answer
    to that is the request llama.cpp does understand, not a failed generation.
    A value that will not convert is dropped for the same reason.
    """
    if not extra:
        return {}
    accepted: dict = {}
    for name, value in extra.items():
        cast = SAMPLING_FIELDS.get(str(name))
        if cast is None or value is None:
            continue
        try:
            accepted[str(name)] = cast(value)
        except (TypeError, ValueError):
            continue
    return accepted


class LlamaClient:
    def __init__(self, base_url: str, api_key: str): self.base_url, self.api_key = base_url.rstrip("/"), api_key

    def stream_chat(self, messages: list[dict], max_tokens: int, seed: int, on_text: Callable[[str], None], cancel: threading.Event | None = None, temperature: float = 0.85, top_p: float = 0.95, extra_sampling: Mapping | None = None) -> str:
        # Writer defaults match upstream backend.chat_stream (0.85 / 0.95); the
        # smart-negative pass overrides them with upstream's cooler 0.3 / 0.9.
        # extra_sampling is Krea's Creativity control and nothing else's: it is
        # empty for every existing caller, and empty at Creativity 0 and 1, so
        # the payload built here is byte-for-byte the payload built before the
        # parameter existed unless somebody has asked for more variation.
        payload = {"model":"prompt-master","messages":messages,"temperature":temperature,"top_p":top_p,"max_tokens":max_tokens,"seed":seed,"stream":True,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":False,"thinking":False}}
        payload.update(sampling_fields(extra_sampling))
        pieces = []
        with httpx.Client(timeout=httpx.Timeout(600, connect=10), headers={"Authorization":f"Bearer {self.api_key}"}) as client:
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for chunk in assistant_chunks(response.iter_lines()):
                    if cancel and cancel.is_set(): return "".join(pieces)
                    pieces.append(chunk); on_text(chunk)
        return "".join(pieces)
