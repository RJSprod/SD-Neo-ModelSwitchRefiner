"""Saved prompt caches: a conversation's llama.cpp slot, kept on disk across restarts.

llama-server keeps what it has read of a prompt -- the key/value cache -- in the
slot that read it, and a reply that continues the conversation reads only what
is new since the last one. That cache lives in the process. A WebUI restart,
Unload, a change of model or device, the projector arriving: each ends the
process, and the next reply reads the whole conversation again from its first
token. From one user's logs, five days on an Intel Arc: eighteen WebUI
restarts, and every first reply after one read four to six thousand tokens at
50 to 90 a second -- a minute or more before its first word, on top of the
model load. On an RTX 5090 the same read took one to two seconds, which is why
the cost only ever showed on the slower device.

llama-server can write a slot's cache to a file and read it back
(``--slot-save-path``, ``POST /slots/{id}?action=save|restore``). The build
this was written against (b10621) saves a slot with pictures in it too -- the
image chunks are serialized beside the tokens -- so a projector loaded from the
start does not rule it out, as it did on older builds. This module uses that:

* **After every completed conversation reply** the slot that answered is saved
  to a file named for the conversation and for the server that wrote it. The
  reply asks llama-server which slot answered (``verbose`` with
  ``response_fields: ["id_slot"]``, so the extra field is that number and
  nothing else), which keeps the file the right slot's with six warm caches
  shared by several roles.
* **Before the first reply of a conversation on a freshly started server** a
  saved file for that conversation and that server is read into a slot nobody
  has used yet, and the reply is sent to that slot. The prompt resumes where
  the file ends, as it would have in the process that wrote it.

What a file is valid for -- its *identity* -- is the llama-server build, the
model file, the projector, whether the sliding-window cache is kept whole, and
the cache types. Anything else that differs either does not matter to a saved
cache or makes llama-server refuse the restore, which costs nothing but the
attempt: the reply reads its prompt in full, as it would have without this,
and the file it saves afterwards is the new server's.

**A model with a sliding window is resumed only where it can be exactly.**
Checked against a real llama-server at b10621, not only read from its source.
When llama.cpp writes a slot, the sliding-window blocks keep only the positions
inside the window -- ``llama_kv_cache::state_write`` skips every cell the
window has passed, and it does so with ``--swa-full`` as well, because the
full-size window cache is still built with the model's window. A restored slot
of such a model holds exactly one window of positions, and:

* **with the window cache** llama.cpp never resumes it. Its check that enough
  of the window is left is inclusive by one, a restored slot has exactly one
  window and no more, and the checkpoint it would fall back on is not part of a
  slot file. Every restore was read in full. So nothing is saved there.
* **with the full cache** it resumes wherever the new prompt parts from the
  saved one within the last window. Where that is the end of the saved state,
  nothing is missing and the resume is exact. Anywhere earlier, the tokens it
  reads again see part of their window missing.

So for a model with a window, on the full cache, a file is read back only after
the new prompt has been shown to *extend* the saved one token for token: the
file's own token list against llama-server's rendering (``/apply-template``)
and tokenizing (``/tokenize``) of the new request, which is how the server
itself will see it. A request that does not extend it -- a regenerated reply,
an edit, a picture in the prompt -- is read in full, as it would have been
without this. Gemma 4 needs :mod:`mc_llm_template` for any of its prompts to
extend the last one, so its cache is saved only when that fix is on the line.

A model without a window -- pure attention, or a hybrid whose recurrent state
llama.cpp either resumes exactly or re-reads -- is saved and restored with no
check, and resumes exactly where a reply in a running server would have.

Nothing here is content. The files hold what llama-server wrote; the names are
hashes; the console lines give token counts, sizes and times, never text, and
never the conversation's name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SLOT_SAVE_FLAG = "--slot-save-path"
"""llama-server's directory for slot files. It must exist when the server starts:
the build refuses to start on a path that is not a directory."""

DIRNAME = "prompt_caches"
"""Under the LLM data root, beside ``logs`` and ``data``."""

SUFFIX = ".slot"

OPT_SAVE = "model_chain_llm_saved_caches"
"""Whether conversations' prompt caches are saved to disk and restored."""

OPT_BUDGET_GB = "model_chain_llm_saved_caches_gb"
"""How much disk the saved caches may take, oldest removed first."""

DEFAULT_BUDGET_GB = 4.0
"""Room for a working set. A saved cache is the conversation's key/value cache,
so its size is the model's cache per token times the conversation's length:
from tens of megabytes for a hybrid model, whose few attention blocks are all
that keep one, to several hundred for a long conversation with a model that
attends in every block."""

SLOTS_TIMEOUT = 10.0
RESTORE_TIMEOUT = 120.0
SAVE_TIMEOUT = 120.0
"""Seconds. A save or restore is a file of a few hundred megabytes read or
written by llama-server, and every request has a deadline."""

SEQUENCE_MAGIC = 0x67677371
"""``ggsq``: llama.cpp's sequence state file (``LLAMA_STATE_SEQ_MAGIC``)."""

SEQUENCE_VERSION = 2
"""``LLAMA_STATE_SEQ_VERSION`` at b10621."""

PACKED_VERSION = 1
"""``SERVER_TOKENS_STATE_VERSION`` at b10621: how the server packs its tokens."""

NULL_TOKEN = -1
"""``LLAMA_TOKEN_NULL``: the packed layout's first word, and a picture's place."""

SLOT_FIELDS = ("verbose", "response_fields")
"""What a reply adds to its request so that its answer names its slot."""

_GB = 1024 ** 3
_MB = 1024 ** 2


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def enabled() -> bool:
    import mc_broker

    value = mc_broker.option(OPT_SAVE, True)
    if isinstance(value, str):
        return value.strip().casefold() not in ("false", "0", "off", "no")
    return bool(value)


def budget_bytes() -> int:
    import mc_broker

    raw = mc_broker.option(OPT_BUDGET_GB, DEFAULT_BUDGET_GB)
    try:
        gigabytes = float(raw)
    except (TypeError, ValueError):
        gigabytes = DEFAULT_BUDGET_GB
    return int(max(gigabytes, 0.0) * _GB)


def active() -> bool:
    """Whether saving and restoring happen now: the setting on and disk to spend."""
    return enabled() and budget_bytes() > 0


def directory() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / DIRNAME


# --------------------------------------------------------------------------- #
# Starting a server that can save
# --------------------------------------------------------------------------- #


def windowed(model) -> bool | None:
    """Whether ``model`` has a sliding window; None when its header cannot be read."""
    import mc_gguf

    described = mc_gguf.describe(model) if model is not None else None
    if described is None:
        return None
    return described.sliding_window > 0 or any(described.swa_blocks)


def resumable(model, full_cache: bool = False, template_fixed: bool = False) -> bool:
    """Whether a saved slot of ``model`` can ever be resumed exactly on this start.

    Always for a model without a sliding window. For one with a window, only on
    the full cache, and -- for a template whose history does not extend its
    last prompt without :mod:`mc_llm_template` -- only with that fix on the
    line. A header that cannot be read is the side that saves nothing.
    """
    has_window = windowed(model)
    if has_window is None:
        return False
    if not has_window:
        return True
    if not full_cache:
        return False
    import mc_llm_template

    return template_fixed or not mc_llm_template.needs_it(model)


def launch_flags(configuration, full_cache: bool = False,
                 template_fixed: bool = False) -> list[str]:
    """``--slot-save-path <folder>`` when saving is on, the model can be resumed
    and the build has the flag.

    ``full_cache`` and ``template_fixed`` are what the rest of this start's
    command line says: whether ``--swa-full`` and ``--chat-template-file`` are on
    it. The folder is made here, before the start, because llama-server refuses
    a path that is not a directory as a bad argument and would not start at
    all. A folder that cannot be made leaves the flag off: the server starts as
    it always did, and nothing is saved.
    """
    if not active() or not resumable(getattr(configuration, "model", None), full_cache,
                                     template_fixed):
        return []
    import mc_llm_runtime

    if not mc_llm_runtime.runtime_supports(SLOT_SAVE_FLAG, configuration):
        return []
    folder = directory()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("Model Chain: could not make the folder for saved prompt caches",
                     exc_info=True)
        return []
    return [SLOT_SAVE_FLAG, str(folder)]


def identity(runtime, model, projector, full_cache: bool, cache_types: str = "",
             template=None) -> str:
    """Which server a saved cache belongs to, as twelve hex digits.

    Each file part is a path, size and modification time, so a replaced build,
    a re-downloaded model, a different projector or a different chat template
    is a different identity and its files are simply never offered to the new
    one. ``template`` is the ``--chat-template-file`` the start was given, if
    any: the template decides every token of every prompt.
    """
    parts = [_stamp(runtime), _stamp(model), _stamp(projector),
             "full" if full_cache else "window", str(cache_types or "default"),
             _stamp(template or None)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _stamp(path) -> str:
    if path is None:
        return "-"
    try:
        found = Path(path)
        stat = found.stat()
        return f"{str(found).casefold()}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        return f"{str(path).casefold()}|?"


def started(process, flags, identity_: str, model=None) -> "Server | None":
    """What saving knows about a server that has just started, or None.

    None unless the start carried :data:`SLOT_SAVE_FLAG`. Read off the flags the
    start was actually given rather than off the setting, because the setting
    can change while the server runs and only what the process was told
    decides what it can do. A model with a sliding window gets a server that
    checks a file before reading it back (see the module docstring).
    """
    flags = [str(flag) for flag in flags or ()]
    if SLOT_SAVE_FLAG not in flags:
        return None
    position = flags.index(SLOT_SAVE_FLAG)
    if position + 1 >= len(flags):
        return None
    port = getattr(process, "port", None)
    key = getattr(process, "api_key", None)
    if not port or not key:
        return None
    return Server(f"http://127.0.0.1:{port}", str(key), Path(flags[position + 1]), identity_,
                  check=bool(windowed(model)) if model is not None else False)


def _conversation_id(conversation) -> str:
    return hashlib.sha256(str(conversation).encode("utf-8")).hexdigest()[:24]


# --------------------------------------------------------------------------- #
# One running server
# --------------------------------------------------------------------------- #


class Refused(RuntimeError):
    """llama-server answered a slot action with an error."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status

    @property
    def permanent(self) -> bool:
        """A request this server will refuse however often it is asked."""
        return 400 <= self.status < 500


class Server:
    """What saving knows about one running llama-server, for as long as it runs."""

    def __init__(self, base_url: str, api_key: str, folder: Path, identity_: str,
                 check: bool = False):
        self.base_url = base_url
        self.api_key = api_key
        self.folder = Path(folder)
        self.identity = identity_
        self.check = check
        """Whether a file is read back only into a prompt that extends it exactly:
        a model with a sliding window, whose saved slot holds one window."""
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        """Conversations this process has answered or been offered a file for.
        A conversation is restored at most once per process: after its first
        reply its cache is in a slot, and after a save the file says the same."""
        self._restored: set[int] = set()
        """Slots a file was read into, so that two conversations are never read
        into the same one."""
        self._saving = threading.Lock()

    def file_name(self, conversation) -> str:
        return f"{_conversation_id(conversation)}-{self.identity}{SUFFIX}"

    # -- restore ---------------------------------------------------------- #

    def restore(self, conversation, body: dict | None = None) -> int | None:
        """Read this conversation's saved cache into an unused slot, and name it.

        None when there is nothing to do or it could not be done: the
        conversation has been answered by this process already, there is no
        file for it, every slot has been used, llama-server refused the file,
        or -- on a server that checks -- the request ``body`` does not extend
        the saved prompt token for token. A refusal of the file itself removes
        it, so a start that cannot use it does not pay for trying again; the
        reply's own save replaces it.
        """
        key = _conversation_id(conversation)
        with self._lock:
            if key in self._seen:
                return None
            self._seen.add(key)
        name = self.file_name(conversation)
        path = self.folder / name
        if not path.is_file():
            return None
        if self.check and not self._extends(path, body):
            return None
        try:
            slot = self._unused_slot()
        except Exception as exc:
            logger.info("Model Chain: could not ask llama-server for a free slot to restore "
                        "this conversation's saved prompt cache into (%s); the prompt is read "
                        "in full", _reason(exc))
            return None
        if slot is None:
            logger.info("Model Chain: every llama-server slot has been used since it started, "
                        "so this conversation's saved prompt cache was not restored; the "
                        "prompt is read in full")
            return None
        began = time.monotonic()
        try:
            answer = self._post(f"/slots/{slot}?action=restore", {"filename": name},
                                RESTORE_TIMEOUT)
        except Refused as refused:
            logger.info("Model Chain: llama-server would not restore this conversation's "
                        "saved prompt cache (%s); the prompt is read in full", refused)
            if refused.permanent:
                _remove(path)
            return None
        except Exception as exc:
            logger.info("Model Chain: could not restore this conversation's saved prompt "
                        "cache (%s); the prompt is read in full", _reason(exc))
            return None
        with self._lock:
            self._restored.add(slot)
        _touch(path)
        tokens = _number(answer, "n_restored")
        read = _number(answer, "n_read")
        logger.info("Model Chain: restored this conversation's saved prompt cache — %s tokens, "
                    "%.0f MB in %.1fs; the reply reads only what is new since it was saved",
                    f"{tokens:,}", read / _MB, time.monotonic() - began)
        return slot

    def _extends(self, path: Path, body: dict | None) -> bool:
        """Whether ``body``'s prompt starts with every token of the saved one.

        The saved tokens are the file's own; the new ones are llama-server's
        rendering of the request and its tokenizing of that, with the special
        tokens added as a chat request adds them. A picture on either side is a
        no: its tokens are not text, and a restore that might not be exact is
        not made.
        """
        saved = saved_tokens(path)
        if not saved:
            logger.info("Model Chain: this conversation's saved prompt cache could not be "
                        "checked against the new prompt, so it was not read back; the prompt "
                        "is read in full")
            return False
        if not isinstance(body, dict) or _has_pictures(body):
            return False
        try:
            rendered = self._post("/apply-template", body, SLOTS_TIMEOUT)
            prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
            if not isinstance(prompt, str):
                return False
            answer = self._post("/tokenize", {"content": prompt, "add_special": True,
                                              "parse_special": True}, SLOTS_TIMEOUT)
            tokens = answer.get("tokens") if isinstance(answer, dict) else None
        except Exception as exc:
            logger.info("Model Chain: could not check this conversation's saved prompt cache "
                        "against the new prompt (%s); the prompt is read in full", _reason(exc))
            return False
        if not isinstance(tokens, list):
            return False
        if len(tokens) > len(saved) and tokens[:len(saved)] == saved:
            return True
        shared = _common(tokens, saved)
        logger.info("Model Chain: this prompt parts from the saved one at token %s of %s, "
                    "before its end, so the saved prompt cache was not read back — with "
                    "a sliding window a resume there would not be exact; the prompt is "
                    "read in full", f"{shared:,}", f"{len(saved):,}")
        return False

    def _unused_slot(self) -> int | None:
        """A slot that is idle and has never run a task, and was not restored into.

        ``id_task`` appears in a slot's entry once it has run one. A slot that
        has is holding somebody's cache -- another role's, or this
        conversation's own -- and reading a file over it would trade one cached
        prompt for another.
        """
        answer = self._get("/slots", SLOTS_TIMEOUT)
        entries = answer if isinstance(answer, list) else []
        with self._lock:
            taken = set(self._restored)
        for entry in sorted((item for item in entries if isinstance(item, dict)),
                            key=lambda item: item.get("id", 0)):
            slot = entry.get("id")
            if not isinstance(slot, int) or slot in taken:
                continue
            if entry.get("is_processing") or "id_task" in entry:
                continue
            return slot
        return None

    # -- save ------------------------------------------------------------- #

    def save_later(self, conversation, slot: int) -> None:
        """Save ``slot`` for ``conversation`` on a thread of its own.

        After the reply rather than inside it: the words have all arrived, and
        a save is llama-server writing a file, which nothing on screen should
        wait for. A reply that follows at once queues behind the save in
        llama-server, for the fraction of a second the write takes.
        """
        key = _conversation_id(conversation)
        with self._lock:
            self._seen.add(key)
        worker = threading.Thread(target=self._save_quietly, args=(conversation, slot),
                                  name="model-chain-save-prompt-cache", daemon=True)
        worker.start()

    def _save_quietly(self, conversation, slot: int) -> None:
        try:
            self.save(conversation, slot)
        except Exception:
            logger.debug("Model Chain: saving a prompt cache failed", exc_info=True)

    def save(self, conversation, slot: int) -> bool:
        name = self.file_name(conversation)
        with self._saving:
            if not active():
                return False
            began = time.monotonic()
            try:
                answer = self._post(f"/slots/{slot}?action=save", {"filename": name},
                                    SAVE_TIMEOUT)
            except Exception as exc:
                logger.info("Model Chain: could not save this conversation's prompt cache "
                            "(%s)", _reason(exc))
                return False
            tokens = _number(answer, "n_saved")
            written = _number(answer, "n_written")
            logger.info("Model Chain: saved this conversation's prompt cache — %s tokens, "
                        "%.0f MB in %.1fs", f"{tokens:,}", written / _MB,
                        time.monotonic() - began)
            prune(self.folder, keep=name)
            return True

    # -- the wire --------------------------------------------------------- #

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get(self, path: str, timeout: float):
        import httpx

        with httpx.Client(timeout=httpx.Timeout(timeout, connect=5),
                          headers=self._headers()) as client:
            response = client.get(f"{self.base_url}{path}")
            return _answer(response)

    def _post(self, path: str, body: dict, timeout: float):
        import httpx

        with httpx.Client(timeout=httpx.Timeout(timeout, connect=5),
                          headers=self._headers()) as client:
            response = client.post(f"{self.base_url}{path}", json=body)
            return _answer(response)


def saved_tokens(path: Path) -> list[int] | None:
    """The token list a slot file was saved with, or None.

    llama.cpp's sequence file: ``ggsq``, version 2, a token count and that many
    32-bit tokens, then the state. The b10621 server packs its own structure
    into those tokens -- a null marker, its version (1), the text tokens, the
    positions of any pictures and the pictures themselves -- and a file with
    pictures in it is None here, as is any layout this does not recognise.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(12)
            if len(head) != 12:
                return None
            magic, version, count = struct.unpack("<III", head)
            if magic != SEQUENCE_MAGIC or version != SEQUENCE_VERSION or count == 0:
                return None
            raw = handle.read(4 * count)
    except OSError:
        return None
    if len(raw) != 4 * count:
        return None
    packed = list(struct.unpack(f"<{count}i", raw))
    if packed[0] != NULL_TOKEN:
        return packed
    if count < 4 or packed[1] != PACKED_VERSION:
        return None
    length = packed[2]
    if length < 0 or 3 + length >= count:
        return None
    tokens = packed[3:3 + length]
    pictures = packed[3 + length]
    if pictures != 0 or NULL_TOKEN in tokens:
        return None
    return tokens


def _has_pictures(body: dict) -> bool:
    for message in body.get("messages") or ():
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, (list, tuple)) and any(
                isinstance(part, dict) and part.get("type") != "text" for part in content):
            return True
    return False


def _common(first: list, second: list) -> int:
    count = 0
    for one, other in zip(first, second):
        if one != other:
            break
        count += 1
    return count


def _answer(response):
    """The JSON of a slot action's answer, or :class:`Refused` with its message."""
    status = int(getattr(response, "status_code", 0) or 0)
    try:
        body = response.json()
    except Exception:
        body = None
    if status >= 400 or status == 0:
        message = ""
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or "")
            elif isinstance(error, str):
                message = error
        raise Refused(message or f"HTTP {status}", status)
    return body


def _number(answer, key: str) -> int:
    try:
        return max(int((answer or {}).get(key) or 0), 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or type(exc).__name__


def _touch(path: Path) -> None:
    try:
        os.utime(path, None)
    except OSError:
        pass


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        logger.debug("Model Chain: could not remove a saved prompt cache", exc_info=True)


# --------------------------------------------------------------------------- #
# The folder
# --------------------------------------------------------------------------- #


def prune(folder: Path, keep: str = "", budget: int | None = None) -> int:
    """Remove the least recently used saved caches beyond the disk budget.

    ``keep`` is the file just written, and it stays whatever it weighs: the
    budget is for the ones that came before it. Everything else is kept newest
    first while the total stays within the budget. A restore counts as a use
    (it touches the file), so the conversations in use are the ones kept.
    """
    budget = budget_bytes() if budget is None else int(budget)
    found = []
    try:
        candidates = list(Path(folder).glob(f"*{SUFFIX}"))
    except OSError:
        return 0
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append((path.name == keep, stat.st_mtime, stat.st_size, path))
    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    total, removed = 0, 0
    for index, (_kept, _mtime, size, path) in enumerate(found):
        if index == 0 or total + size <= budget:
            total += size
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            logger.debug("Model Chain: could not remove a saved prompt cache", exc_info=True)
    if removed:
        logger.info("Model Chain: removed %d saved prompt cache%s, the least recently used, "
                    "to stay within %.1f GB", removed, "" if removed == 1 else "s",
                    budget / _GB)
    return removed


def usage() -> tuple[int, int]:
    """``(files, bytes)`` of saved caches on disk, for a status line."""
    count, size = 0, 0
    try:
        for path in directory().glob(f"*{SUFFIX}"):
            try:
                size += path.stat().st_size
                count += 1
            except OSError:
                continue
    except OSError:
        pass
    return count, size


# --------------------------------------------------------------------------- #
# The client that saves
# --------------------------------------------------------------------------- #


def slot_fields(pinned: int | None) -> dict:
    """What a conversation reply adds to its request body, and nothing else.

    ``verbose`` makes llama-server put its own view of the request beside the
    OpenAI-shaped answer, and ``response_fields`` cuts that view down to the
    slot number -- without it the view carries the whole prompt back.
    ``id_slot`` is sent only for the reply that follows a restore, to the slot
    the file was read into.
    """
    fields = {"verbose": True, "response_fields": ["id_slot"]}
    if pinned is not None:
        fields["id_slot"] = int(pinned)
    return fields


def watching(lines, found: dict):
    """``lines``, unchanged, noting the slot any of them names into ``found``."""
    for line in lines:
        if "__verbose" in line:
            text = line.strip()
            if text.startswith("data:"):
                try:
                    payload = json.loads(text[5:].strip())
                except ValueError:
                    payload = None
                verbose = payload.get("__verbose") if isinstance(payload, dict) else None
                slot = verbose.get("id_slot") if isinstance(verbose, dict) else None
                if isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0:
                    found["slot"] = slot
        yield line


_client_class = None
_client_class_lock = threading.Lock()


def client(base_url: str, api_key: str, sampling, server: Server):
    """A :class:`LlamaClient` that can also save and restore a conversation.

    Defined on first use rather than at import, because nothing in this
    extension imports the vendored package before the language model is
    actually wanted (a failure in the LLM half must not reach a WebUI that
    never opens it).
    """
    global _client_class
    with _client_class_lock:
        if _client_class is None:
            _client_class = _define_client()
    return _client_class(base_url, api_key, sampling, server)


def _define_client():
    import httpx
    from prompt_master.inference.llama_client import LlamaClient
    from prompt_master.inference.local_only import check_messages
    from prompt_master.inference.sse import assistant_chunks

    class SavingClient(LlamaClient):
        """The vendored client, plus one method a conversation reply calls.

        ``stream_chat`` is inherited untouched and is what every other caller
        still uses. :meth:`stream_conversation` builds the same request body --
        :func:`request_body` is checked against the vendored one by a test, so
        the two cannot drift apart unnoticed -- with :func:`slot_fields` added.
        """

        def __init__(self, base_url, api_key, sampling, server: Server):
            super().__init__(base_url, api_key, sampling)
            self.saved_caches = server

        def request_body(self, messages, max_tokens, seed, temperature, top_p) -> dict:
            payload = {"model": "prompt-master", "messages": messages,
                       "temperature": temperature, "top_p": top_p,
                       "max_tokens": max_tokens, "seed": seed, "stream": True,
                       "reasoning_effort": "none",
                       "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
            payload.update(self.sampling)
            return payload

        def stream_conversation(self, conversation, messages, max_tokens, seed, on_text,
                                cancel=None, temperature=0.85, top_p=0.95) -> str:
            server = self.saved_caches
            if not conversation or server is None or not active():
                return self.stream_chat(messages, max_tokens, seed, on_text, cancel,
                                        temperature=temperature, top_p=top_p)
            check_messages(messages)
            payload = self.request_body(messages, max_tokens, seed, temperature, top_p)
            pinned = server.restore(conversation, dict(payload))
            payload.update(slot_fields(pinned))
            found: dict = {}
            pieces = []
            with httpx.Client(timeout=httpx.Timeout(600, connect=10),
                              headers={"Authorization": f"Bearer {self.api_key}"}) as http:
                with http.stream("POST", f"{self.base_url}/v1/chat/completions",
                                 json=payload) as response:
                    response.raise_for_status()
                    for chunk in assistant_chunks(watching(response.iter_lines(), found)):
                        if cancel and cancel.is_set():
                            return "".join(pieces)
                        pieces.append(chunk)
                        on_text(chunk)
            if cancel is not None and cancel.is_set():
                return "".join(pieces)
            slot = found.get("slot", pinned)
            if slot is not None:
                server.save_later(conversation, slot)
            return "".join(pieces)

    return SavingClient


__all__ = ["DEFAULT_BUDGET_GB", "OPT_BUDGET_GB", "OPT_SAVE", "SLOT_SAVE_FLAG", "Refused",
           "Server", "active", "budget_bytes", "client", "directory", "enabled", "identity",
           "launch_flags", "prune", "resumable", "saved_tokens", "slot_fields", "started",
           "usage", "watching", "windowed"]
