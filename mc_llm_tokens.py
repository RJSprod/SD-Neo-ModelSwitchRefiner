"""Counting a conversation in the model's own tokens.

The prompt builder trimmed a conversation against an estimate -- 3.2 characters
a token, pessimistic on purpose -- because it had no tokenizer. From the user's
logs (2026-11-09) a long role-play thread on Gemma 4 runs at about 4.1
characters a token, so the estimate kept about a fifth less of the conversation
than the context had room for. llama-server has the model's tokenizer and
answers ``POST /tokenize`` from its HTTP thread, without waiting for a slot or
the card, so a running server is asked: a message's text exactly
(:meth:`Counter.text`), and a whole request exactly, rendered by the server's
own chat template first (:meth:`Counter.prompt`) -- which is what the server
itself will count.

Nothing here starts a server. With none running the count is an estimate:
the thread's own measured characters-a-token when it has one (the
``ratio`` :mod:`mc_llm_window` keeps per thread, written whenever a count was
exact), else the builder's 3.2. Every answer says which it is (``exact``).

A message's text is counted once per server and model: counts are remembered
by a digest of the text, never the text itself, and forgotten oldest first.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import OrderedDict

logger = logging.getLogger("model_chain")

ESTIMATE_RATIO = 1 / 3.2
"""Tokens a character, unmeasured: the builder's own pessimistic estimate
(``prompt_master.chat.prompt.CHARS_PER_TOKEN``)."""

MESSAGE_OVERHEAD = 8
"""Tokens a message costs beyond its text: the chat template's turn markers
(Gemma 4's are ``<|turn>user`` and ``<turn|>`` with their line breaks, and a
reply's empty thought marker). A cut is made between whole messages, so this
only decides which message a cut lands after, never what a total says."""

KEEP = 8192
"""Counts remembered, oldest forgotten first."""

TIMEOUT = 5.0

PIECE_CONTEXT = "*"
"""What a word is read after when its pieces are asked for (:meth:`Counter.pieces`).

A tokenizer that puts a space before the start of a text (SentencePiece's own,
``add_space_prefix``) reads `` rain`` alone as two spaces and a word; after
another token it reads it as it does inside a reply. An asterisk is a token of
its own before a word in every vocabulary this was read against, with or
without a space after it, so it never joins onto the word that follows."""

_counts: OrderedDict = OrderedDict()
_pieces: OrderedDict = OrderedDict()
_lock = threading.Lock()


def _digest(server: str, text: str) -> str:
    return hashlib.sha1(f"{server}\0{text}".encode("utf-8", "replace")).hexdigest()


def _remember(key: str, count: int) -> None:
    with _lock:
        _counts[key] = count
        _counts.move_to_end(key)
        while len(_counts) > KEEP:
            _counts.popitem(last=False)


def _recall(key: str):
    with _lock:
        found = _counts.get(key)
        if found is not None:
            _counts.move_to_end(key)
        return found


def forget() -> None:
    """Forget every count. For the tests."""
    with _lock:
        _counts.clear()
        _pieces.clear()


def _piece_text(piece) -> str:
    """A piece as llama-server sends it: text, or its bytes when they are not UTF-8."""
    if isinstance(piece, str):
        return piece
    if isinstance(piece, list):
        try:
            return bytes(int(item) & 0xFF for item in piece).decode("utf-8", "replace")
        except (TypeError, ValueError):
            return ""
    return ""


def estimate(text: str, ratio: float = 0.0) -> int:
    """``text``'s tokens at ``ratio`` tokens a character (the default's when 0)."""
    chars = len(text or "")
    if not chars:
        return 0
    used = ratio if ratio and ratio > 0 else ESTIMATE_RATIO
    return int(math.ceil(chars * used))


class Counter:
    """Counts against one running server, or estimates without one.

    ``client`` is the language model's client for the server that is up (its
    ``base_url`` and ``api_key``), or None. ``model`` names what the server
    loaded, so a count is never taken for another model's. ``ratio`` is the
    thread's measured tokens a character, for estimating.
    """

    def __init__(self, client=None, model: str = "", ratio: float = 0.0):
        self.client = client
        self.server = f"{getattr(client, 'base_url', '')}|{model}" if client is not None else ""
        self.ratio = ratio if ratio and ratio > 0 else 0.0
        self.exact = client is not None
        self.failed = False

    # -- the server -------------------------------------------------------- #

    def _post(self, route: str, body: dict):
        import httpx

        with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=2.0),
                          headers={"Authorization": f"Bearer {self.client.api_key}"}) as http:
            response = http.post(f"{self.client.base_url}{route}", json=body)
            response.raise_for_status()
            return response.json()

    def _tokenized(self, content: str, special: bool) -> int | None:
        try:
            answer = self._post("/tokenize", {"content": content, "add_special": special,
                                              "parse_special": True})
        except Exception as exc:
            self._lost(exc)
            return None
        tokens = answer.get("tokens") if isinstance(answer, dict) else None
        return len(tokens) if isinstance(tokens, list) else None

    def _lost(self, exc) -> None:
        """The server stopped answering: estimate from here on, and say so once."""
        if not self.failed:
            logger.debug("Model Chain: llama-server could not count tokens (%s); estimating",
                         type(exc).__name__)
        self.failed = True
        self.exact = False
        self.client = None

    # -- counting ---------------------------------------------------------- #

    def text(self, text: str) -> int:
        """``text``'s tokens, as the model's tokenizer reads it inside a prompt."""
        text = text or ""
        if not text:
            return 0
        if self.client is None:
            return estimate(text, self.ratio)
        key = _digest(self.server, text)
        found = _recall(key)
        if found is not None:
            return found
        counted = self._tokenized(text, special=False)
        if counted is None:
            return estimate(text, self.ratio)
        _remember(key, counted)
        return counted

    def _tokens_of(self, content: str):
        """``[(id, piece), ...]`` for ``content`` read as plain text, or None."""
        try:
            answer = self._post("/tokenize", {"content": content, "add_special": False,
                                              "parse_special": False, "with_pieces": True})
        except Exception as exc:
            self._lost(exc)
            return None
        tokens = answer.get("tokens") if isinstance(answer, dict) else None
        if not isinstance(tokens, list):
            return None
        found = []
        for token in tokens:
            if not isinstance(token, dict) or isinstance(token.get("id"), bool) \
                    or not isinstance(token.get("id"), int):
                return None
            found.append((int(token["id"]), _piece_text(token.get("piece"))))
        return found

    def pieces(self, text: str):
        """``text``'s tokens as the model reads them after another word, as
        ``[(id, piece), ...]``; None with no server, or when it cannot say.

        Read after :data:`PIECE_CONTEXT` and taken off it, so a tokenizer that
        puts a space before the start of a text answers for the word as it is
        inside a reply. Remembered per server and model, by a digest of the text.
        """
        text = text or ""
        if self.client is None or not text:
            return None
        key = _digest(self.server, "\0pieces\0" + text)
        with _lock:
            found = _pieces.get(key)
            if found is not None:
                _pieces.move_to_end(key)
                return list(found)
        whole = self._tokens_of(PIECE_CONTEXT + text)
        lead = self._tokens_of(PIECE_CONTEXT) if whole is not None else None
        if whole is None or lead is None:
            return None
        if lead and whole[:len(lead)] == lead:
            found = whole[len(lead):]
        else:
            found = self._tokens_of(text)
            if found is None:
                return None
        with _lock:
            _pieces[key] = tuple(found)
            _pieces.move_to_end(key)
            while len(_pieces) > KEEP:
                _pieces.popitem(last=False)
        return list(found)

    def message(self, text: str, still: bool = False) -> int:
        """One message's cost in a prompt: its text, its turn markers, and a
        picture it carries as a still."""
        from prompt_master.chat.prompt import IMAGE_TOKENS

        return self.text(text) + MESSAGE_OVERHEAD + (IMAGE_TOKENS if still else 0)

    def prompt(self, wire) -> tuple[int, bool]:
        """A whole request's tokens, and whether that is exact.

        Rendered by the server's own chat template and tokenized as a chat
        request is, special tokens added. A picture is not text: it is taken
        out before rendering and charged as the builder charges a still.
        """
        from prompt_master.chat.prompt import IMAGE_TOKENS

        pictures = 0
        plain = []
        for message in wire or ():
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                words = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        pictures += 1
                    elif part.get("type") == "text":
                        words.append(str(part.get("text") or ""))
                content = "\n".join(words)
            plain.append({"role": message.get("role"), "content": str(content or "")})
        extra = pictures * IMAGE_TOKENS
        if self.client is not None:
            try:
                rendered = self._post("/apply-template", {
                    "messages": plain, "reasoning_effort": "none",
                    "chat_template_kwargs": {"enable_thinking": False, "thinking": False}})
                prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
            except Exception as exc:
                self._lost(exc)
                prompt = None
            if isinstance(prompt, str):
                counted = self._tokenized(prompt, special=True)
                if counted is not None:
                    return counted + extra, True
        total = sum(self.message(item["content"]) for item in plain)
        return total + extra, False


def for_server(ratio: float = 0.0) -> Counter:
    """A counter for the language model's server that is running, or one that
    estimates when none is. Starts nothing."""
    client = None
    try:
        import mc_llm_sessions as sessions

        runtime = sessions._runtime_for("").runtime
        warm = getattr(runtime, "warm_client", None)
        client = warm(False) if callable(warm) else None
    except Exception:
        logger.debug("Model Chain: could not ask whether the language model is running",
                     exc_info=True)
        client = None
    return for_client(client, ratio)


def for_client(client, ratio: float = 0.0) -> Counter:
    """A counter on ``client``'s server -- the one a request is about to go to --
    or one that estimates when ``client`` is None."""
    model = ""
    if client is not None:
        try:
            import mc_llm_runtime

            model = str(getattr(mc_llm_runtime.config(), "model", "") or "")
        except Exception:
            logger.debug("Model Chain: could not read which model is configured",
                         exc_info=True)
    return Counter(client, model=hashlib.sha1(model.encode("utf-8", "replace")).hexdigest()
                   if model else "", ratio=ratio)


__all__ = ["Counter", "ESTIMATE_RATIO", "MESSAGE_OVERHEAD", "PIECE_CONTEXT", "estimate",
           "for_client", "for_server", "forget"]
