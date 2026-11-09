"""A thread's context window: which of its messages the model is sent.

llama-server keeps a conversation's prompt cached and reads only what is new --
as long as the prompt still begins the way the cached one did. Once a thread is
longer than the context has room for, its oldest messages have to go, and the
moment the first message sent changes, everything after the system prompt is
read again. From the user's logs (2026-11-09) that was 3,300 tokens at about
100 a second on the Intel Arc -- 33 seconds before the first word, on about
every second message of a long thread, because the old trimmer moved the front
a quarter of an estimated window at a time and that thread's replies run 2,000
characters. "i get a lot of delay still between reply ... why did it take so
long between my submitted message and time to first token?"

What changed, and where each part lives:

* **The window's start is kept** (:class:`Window`, ``window.json`` beside the
  threads, never in the conversation's own file, for the emotion's reason: a
  write there is a new revision). The builder no longer works the front out
  afresh for every request; it starts where the thread's window starts, and
  moves that start only when the window no longer fits (:func:`plan`). So a
  restart, a Regenerate or an edit near the end sends the window the cache
  holds.
* **It moves after a reply, not before your message.** When the next message
  would not fit, the start is moved as soon as the reply is saved
  (:func:`lookahead`) and the conversation is read again in the background
  (``mc_llm_conversation_ops``, the cache update), while the reply is being
  read aloud. A message of yours then reads only itself.
* **It is counted in the model's tokens** (:mod:`mc_llm_tokens`), not at 3.2
  characters a token, so the window holds what the context has room for.
* **The room kept for the reply follows the thread's replies**
  (:func:`reserve`): a little more than the longest of its last few, never more
  than its Reply tokens. Oobabooga keeps 512 by default and lets a reply run on
  into what is left; a thread whose replies are 500 tokens was keeping two or
  three thousand for one.
* **You can move it yourself**, between whole messages and never into the last
  exchange: keep the latest so many tokens (:func:`keep_latest`, *Trim*), or
  have the model summarize what goes into a hidden summary sent after the
  system prompt in its place (:mod:`mc_llm_compact`, *Compact*).

When the start does have to move by itself, it moves far enough that the window
is three quarters of the room (:data:`STEP`), so it does not move again for
several messages.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("model_chain")

FILE_NAME = "window.json"
"""Under the LLM data root, beside ``generation.json`` and ``emotions.json``."""

VERSION = 1

STEP = 0.25
"""When the window has to move by itself, it is cut to (1 - STEP) of the room,
so the next several messages fit without another move."""

RESERVE_FLOOR = 256
"""The least room kept for a reply, in tokens."""

RESERVE_NEW = 512
"""The room kept for a reply in a thread with no reply yet: Oobabooga's
default ``max_new_tokens``."""

RESERVE_MARGIN = 1.25
"""The room kept for a reply is this much more than the longest recent reply."""

RECENT_REPLIES = 5

ALLOWANCE_FLOOR = 192
"""Tokens kept free for your next message when deciding, after a reply, whether
the window has to move before it -- at least this, or twice the longest of
your last few messages."""

SUMMARY_HEAD = "[What happened earlier in this conversation]"
"""The line a compacted thread's summary is sent under, at the end of the system
prompt and before the thread's emotion."""

_lock = threading.Lock()


@dataclass
class Window:
    """One thread's window.

    ``front`` is the first message sent. ``summary`` stands in for the messages
    before ``upto``. ``ratio`` is the thread's own tokens a character, from the
    last exact count, for estimating while no server is running.
    """

    front: int = 0
    summary: str = ""
    upto: int = 0
    summary_tokens: int = 0
    ratio: float = 0.0

    def to_dict(self) -> dict:
        found = {}
        if self.front:
            found["front"] = int(self.front)
        if self.summary:
            found.update(summary=self.summary, upto=int(self.upto),
                         summary_tokens=int(self.summary_tokens))
        if self.ratio:
            found["ratio"] = round(float(self.ratio), 5)
        return found

    @classmethod
    def from_dict(cls, data) -> "Window":
        if not isinstance(data, dict):
            return cls()

        def whole(name):
            value = data.get(name)
            return max(int(value), 0) if isinstance(value, (int, float)) \
                and not isinstance(value, bool) else 0

        summary = data.get("summary") if isinstance(data.get("summary"), str) else ""
        ratio = data.get("ratio")
        ratio = float(ratio) if isinstance(ratio, (int, float)) \
            and not isinstance(ratio, bool) and 0 < ratio < 4 else 0.0
        return cls(front=whole("front"), summary=summary.strip(),
                   upto=whole("upto") if summary.strip() else 0,
                   summary_tokens=whole("summary_tokens") if summary.strip() else 0,
                   ratio=ratio)


# -- the file ---------------------------------------------------------------- #


def path() -> Path:
    import mc_llm_paths

    return mc_llm_paths.data_root() / FILE_NAME


def _key(character, thread_id) -> str:
    return f"{str(character or '').strip()}/{str(thread_id or '').strip()}"


def _read() -> dict:
    where = path()
    if not where.is_file():
        return {}
    try:
        from prompt_master.core.config import read_json

        data = read_json(where)
    except Exception:
        logger.warning("Model Chain: the threads' context windows could not be read; each "
                       "thread starts from its newest messages that fit until one is set "
                       "again")
        return {}
    threads = data.get("threads") if isinstance(data, dict) else None
    return dict(threads) if isinstance(threads, dict) else {}


def _write(threads: dict) -> None:
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(path(), {"version": VERSION, "threads": threads})


def for_thread(character, thread_id) -> Window:
    """This thread's window; the default one for a thread that has none."""
    with _lock:
        return Window.from_dict(_read().get(_key(character, thread_id)))


def _update(character, thread_id, change) -> Window:
    """Read, change and write one thread's window under the lock. Never raises
    for a thread with no name; answers the window as kept."""
    if not str(character or "").strip() or not str(thread_id or "").strip():
        return Window()
    name = _key(character, thread_id)
    with _lock:
        threads = _read()
        window = Window.from_dict(threads.get(name))
        before = window.to_dict()
        change(window)
        after = window.to_dict()
        if after != before:
            if after:
                threads[name] = after
            else:
                threads.pop(name, None)
            _write(threads)
        return window


def set_front(character, thread_id, front: int) -> Window:
    """The thread's window starts at message ``front`` from now on."""
    def change(window):
        window.front = max(int(front), 0)

    return _update(character, thread_id, change)


def set_summary(character, thread_id, summary: str, upto: int, tokens: int) -> Window:
    """``summary`` stands in for the messages before ``upto``, and the window
    starts there."""
    def change(window):
        window.summary = str(summary or "").strip()
        window.upto = max(int(upto), 0) if window.summary else 0
        window.summary_tokens = max(int(tokens), 0) if window.summary else 0
        window.front = window.upto

    return _update(character, thread_id, change)


def clear(character, thread_id) -> Window:
    """No summary, and the window back to the newest messages that fit. The
    measured ratio is kept: it is about the thread's words, not its window."""
    def change(window):
        window.front, window.summary, window.upto, window.summary_tokens = 0, "", 0, 0

    return _update(character, thread_id, change)


def remember_ratio(character, thread_id, ratio: float) -> None:
    """Keep the thread's measured tokens a character, when it moved enough to
    matter. Never fails a caller."""
    try:
        if not ratio or ratio <= 0 or ratio >= 4:
            return
        current = for_thread(character, thread_id).ratio
        if current and abs(current - ratio) / current < 0.02:
            return

        def change(window):
            window.ratio = float(ratio)

        _update(character, thread_id, change)
    except Exception:
        logger.debug("Model Chain: could not keep a thread's measured token ratio",
                     exc_info=True)


def carry(character, source_thread, made_thread, index) -> None:
    """Give a branch -- the source's messages through ``index`` -- what of the
    source's window still describes it. Never fails a caller."""
    try:
        source = for_thread(character, source_thread)
        if source == Window():
            return
        last = int(index) if isinstance(index, int) else -1
        made = Window(ratio=source.ratio)
        if source.summary and source.upto <= last + 1:
            made.summary, made.upto = source.summary, source.upto
            made.summary_tokens = source.summary_tokens
        if source.front <= last:
            made.front = max(source.front, made.upto)
        else:
            made.front = made.upto

        def change(window):
            window.front, window.summary = made.front, made.summary
            window.upto, window.summary_tokens = made.upto, made.summary_tokens
            window.ratio = made.ratio

        _update(character, made_thread, change)
    except Exception:
        logger.debug("Model Chain: could not carry a thread's context window", exc_info=True)


def removed(character, thread_id, index: int, count: int, remaining: int) -> None:
    """Messages ``index`` to ``index + count`` were deleted, ``remaining`` are left.

    The window keeps pointing at the same messages. A summary that described
    messages deleted from the end -- Delete from here, into what it covers --
    describes a conversation that did not happen, and goes. Never fails a
    caller.
    """
    try:
        index, count = max(int(index), 0), max(int(count), 0)
        if not count:
            return

        def change(window):
            to_end = index + count >= remaining + count
            if window.summary and index < window.upto:
                if to_end:
                    window.front, window.summary, window.upto = 0, "", 0
                    window.summary_tokens = 0
                    return
                window.upto -= min(count, window.upto - index)
            if index < window.front:
                window.front -= min(count, window.front - index)
            if window.summary:
                window.front = max(window.front, window.upto)

        _update(character, thread_id, change)
    except Exception:
        logger.debug("Model Chain: could not keep a thread's window after a delete",
                     exc_info=True)


def forget(character, thread_id) -> None:
    """Drop a deleted thread's window. Never fails a caller."""
    try:
        name = _key(character, thread_id)
        with _lock:
            threads = _read()
            if name in threads:
                del threads[name]
                _write(threads)
    except Exception:
        logger.debug("Model Chain: could not forget a deleted thread's window", exc_info=True)


# -- what is sent -------------------------------------------------------------- #


def summary_block(window: Window) -> str:
    """What the system prompt ends with for a compacted thread, or ``""``."""
    if not window.summary:
        return ""
    return f"{SUMMARY_HEAD}\n{window.summary}"


def floor(window: Window) -> int:
    """The first message that can be sent: the one after the summary, if any."""
    return window.upto if window.summary else 0


def last_kept(history) -> int:
    """The latest a window may start: your last message, so the last exchange is
    always sent whole. The last message for a thread with none of yours."""
    from prompt_master.chat.history import USER

    for index in range(len(history) - 1, -1, -1):
        if history[index].role == USER:
            return index
    return max(len(history) - 1, 0)


def start_of(window: Window, history) -> int:
    """Where this history's window starts, as kept, and kept within it."""
    if not history:
        return 0
    return min(max(window.front, floor(window)), last_kept(history))


def stills(history, every_picture: bool = False) -> set:
    """Which messages are sent with their picture: the newest picture, or every
    one (``prompt_master.chat.prompt.stills_carried``)."""
    from prompt_master.chat.prompt import stills_carried

    pictured = [index for index, message in enumerate(history) if message.image]
    return set(pictured[len(pictured) - stills_carried(len(pictured), every_picture):])


def costs(history, counter, every_picture: bool = False, start: int = 0) -> list:
    """Every message's tokens, from ``start``; 0 before it, where nothing is sent."""
    carried = stills(history, every_picture)
    found = [0] * len(history)
    for index in range(max(start, 0), len(history)):
        message = history[index]
        found[index] = counter.message(message.text, still=index in carried)
    return found


def reserve(history, reply_limit: int, counter) -> int:
    """The room kept for the reply: a little more than the longest of the
    thread's last few replies, never more than ``reply_limit``."""
    from prompt_master.chat.history import ASSISTANT

    limit = max(int(reply_limit or 0), 1)
    replies = [message for message in history if message.role == ASSISTANT]
    replies = replies[-RECENT_REPLIES:]
    if not replies:
        return min(limit, RESERVE_NEW)
    longest = max(counter.text(message.text) for message in replies)
    return min(limit, max(RESERVE_FLOOR, int(math.ceil(longest * RESERVE_MARGIN))))


def allowance(history, counter) -> int:
    """Tokens kept free for the next message of yours."""
    from prompt_master.chat.history import USER

    yours = [message for message in history if message.role == USER][-3:]
    longest = max((counter.text(message.text) for message in yours), default=0)
    return max(ALLOWANCE_FLOOR, 2 * longest)


@dataclass
class Plan:
    """What one request sends: the window's start, and what it costs."""

    front: int
    total: int
    room: int
    reserve: int
    context: int
    moved: bool = False
    exact: bool = False
    costs: list = field(default_factory=list)


def _cut(spent: list, front: int, fixed: int, target: float, last: int) -> tuple[int, int]:
    total = fixed + sum(spent[front:])
    while front < last and total > target:
        total -= spent[front]
        front += 1
    return front, total


def plan(history, window: Window, counter, fixed: int, context_size: int,
         reply_limit: int, every_picture: bool = False, ahead: int = 0) -> Plan:
    """Where this request's window starts, and what it costs.

    ``fixed`` is what every request in the thread sends besides its messages:
    the system prompt with the summary and the emotion, and an instruction.
    ``ahead`` is room kept for a message that is not in ``history`` yet
    (:func:`lookahead`). The window starts where it is kept; only when it no
    longer fits does it move -- far enough that it fills ``1 - STEP`` of the
    room -- and never past your last message.
    """
    context = max(int(context_size or 0), 1)
    kept = reserve(history, reply_limit, counter)
    room = max(context - kept, 1)
    front = start_of(window, history)
    spent = costs(history, counter, every_picture, start=front)
    total = fixed + sum(spent[front:])
    moved = False
    if total + ahead > room:
        front, total = _cut(spent, front, fixed + ahead, room * (1 - STEP),
                            last_kept(history))
        total -= ahead
        moved = front != start_of(window, history)
    return Plan(front=front, total=total, room=room, reserve=kept, context=context,
                moved=moved, exact=counter.exact, costs=spent)


def lookahead(history, window: Window, counter, fixed: int, context_size: int,
              reply_limit: int, every_picture: bool = False) -> Plan:
    """:func:`plan` after a reply: would your next message still fit? If not, the
    window moves now -- before you have written it."""
    return plan(history, window, counter, fixed, context_size, reply_limit, every_picture,
                ahead=allowance(history, counter))


def keep_latest(history, window: Window, counter, keep_tokens: int,
                every_picture: bool = False) -> tuple[int, int]:
    """Where a window keeping the latest ``keep_tokens`` of messages starts, and
    what they cost: between whole messages, after any summary, and never into
    the last exchange. Further back than the window starts now, when asked."""
    if not history:
        return 0, 0
    first = floor(window)
    spent = costs(history, counter, every_picture, start=first)
    last = last_kept(history)
    front = len(history)
    kept = 0
    while front > first and kept + spent[front - 1] <= max(int(keep_tokens), 0):
        kept += spent[front - 1]
        front -= 1
    # And the last exchange whatever it costs.
    front = min(max(front, first), last)
    return front, sum(spent[front:])


__all__ = ["ALLOWANCE_FLOOR", "Plan", "RESERVE_FLOOR", "RESERVE_NEW", "STEP", "SUMMARY_HEAD",
           "Window", "allowance", "carry", "clear", "costs", "floor", "for_thread", "forget",
           "keep_latest", "last_kept", "lookahead", "plan", "remember_ratio", "removed",
           "reserve", "set_front", "set_summary", "start_of", "summary_block"]
