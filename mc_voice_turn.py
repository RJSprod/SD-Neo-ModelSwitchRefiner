"""One assistant reply, on its way to a speaker, with a name of its own.

A VoiceTurn is the server's record of a reply that is *eligible to be spoken*.
It exists because streaming speech has three producers and two consumers that
all have to agree about which reply they are working on:

    the LLM generator            appends text as it arrives
    the segmenter                decides what is whole enough to say
    the Voice Worker             turns segments into PCM
    the HTTP stream              hands PCM to one browser
    Stop, the microphone,
    Auto Speak and unload        cancel all of it

Before streaming, "which reply" was answered by a one-shot token that stood for
a finished string; the reply was over before anything else started, so nothing
could race. Now the reply is still being written while its opening is being
spoken, and section 24 asks for one identity that follows the whole chain --
generator to segment to worker frame to PCM block to the browser's scheduled
audio source -- so that audio from a cancelled turn cannot be played over the
reply that replaced it. That identity is :attr:`VoiceTurn.id`, it is opaque,
and it is the only thing about a turn the browser ever sees.

What a turn holds, and what it refuses to hold
----------------------------------------------
It holds the reply text on its way through the segmenter, PCM on its way to one
HTTP response, and counters. It writes nothing, logs no text and no audio, and
the numbers it keeps are the content-free ones section 36 permits: character
counts, timestamps, segment counts, seconds of audio, a cancellation category.
:func:`VoiceTurn.metrics` is what may be logged, and it is built by naming the
permitted fields rather than by removing the forbidden ones.

Bounded, everywhere
-------------------
Section 23. The text backlog is bounded by the same speech ceiling the
completed-reply path already had, and by a segment count. The audio queue is
bounded in seconds. Neither bound is enforced by dropping: text past the
ceiling stops the turn with a visible warning (section 11 -- no silent
truncation), and a full audio queue applies backpressure that reaches all the
way to the sherpa callback through the pipe, which is where it belongs.

Cancellation
------------
:attr:`VoiceTurn.cancelled` is a ``threading.Event`` and it is the only thing
any code has to look at to know whether to stop. Every wait in this module is
either on that event or on a queue with a short timeout in a loop that checks
it, so there is no state in which a turn can be waiting for something that will
never arrive and not notice that it has been cancelled. Cancelling twice, or
cancelling a turn that has already finished, is defined and harmless: section 26
requires Stop to be idempotent because two paths deliberately press it.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time

import mc_voice_segment as segment

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

MAX_SOURCE_CHARS = 60_000
"""The speech ceiling for one turn, in characters of source text.

The completed-reply path measured its own ceiling in encoded bytes and refused
past it with a sentence rather than speaking half an answer. Streaming keeps
the policy and states it in characters, because that is the unit the segmenter
and the backlog are both counted in. Sixty thousand characters is about an hour
of speech: past it, something has gone wrong with a generation rather than
somebody having written a long reply.
"""

MAX_BACKLOG_SEGMENTS = 512
"""How many committed-but-unsynthesized segments may wait.

Reached only when synthesis has fallen a very long way behind generation, which
is the same condition :data:`MAX_SOURCE_CHARS` describes from the other side.
Both are refusals rather than drops.
"""

AUDIO_QUEUE_SECONDS = 18.0
"""How much generated speech may wait in this process for one browser.

Section 23's 15-20 seconds. Past it the queue stops accepting, the pipe from
the worker fills, the worker's writer blocks, its own queue fills and the
sherpa callback stops being called -- backpressure that reaches the producer
without a single frame being dropped.
"""

PUMP_POLL = 0.05
"""How often a blocked wait looks at the cancellation event. Fifty milliseconds
is below the hundred section 29 asks for and is not a busy loop."""

CLIENT_WAIT = 30.0
"""How long a turn waits for a browser to open its stream before giving up.

Generous: the page has to receive the turn id through Gradio and open a fetch,
which on a phone over a mesh VPN is not instant. Bounded: a turn nobody is
listening to must not keep the inference lane.
"""

FIRST_SEGMENT_WAIT = 120.0
"""How long a turn with no text at all stays alive before giving up.

A generation that produces nothing for two minutes is a generation that has
failed in a way this module cannot see, and a turn that waits forever is a
thread and a queue that outlive the conversation they belonged to.
"""


class VoiceTurnError(RuntimeError):
    """A turn that could not be spoken. Never fatal to Conversation."""


# --------------------------------------------------------------------------- #
# The turn
# --------------------------------------------------------------------------- #


class VoiceTurn:
    """One reply's speech, from the first chunk to the last sample."""

    def __init__(self, voice_id: str = "", sid: int = 0, labels=(), page: str = "",
                 speaker=None, max_source_chars: int = MAX_SOURCE_CHARS, profile=None):
        self.id = secrets.token_urlsafe(18)
        self.page = str(page or "")
        self.voice_id = str(voice_id or "")
        self.sid = int(sid or 0)
        self.profile = dict(profile) if profile else None
        """The delivery this reply is spoken with, resolved when the turn was
        created. Held rather than read at ``begin_turn`` for the reason the
        voice is: what a reply sounds like is decided once, at its beginning,
        and a slider moved while it is speaking changes the next one."""
        self.created_at = time.monotonic()

        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.started = threading.Event()
        """Set when the first PCM frame has been accepted, so a caller can tell
        "nothing has been spoken yet" from "speech is under way" -- which is the
        difference between a cancel that can be silent and one that cannot."""

        self.attached = threading.Event()
        """Set when a browser has opened the stream for this turn.

        Synthesis starts as soon as the first segment is whole rather than
        waiting for a listener, because waiting would give away the latency the
        whole feature is for. The cost is that a turn nobody ever connects to
        would fill its audio queue and then hold the worker's one inference
        lane against a bounded queue forever -- so :data:`CLIENT_WAIT` is how
        long that is allowed to go on.
        """

        self._lock = threading.Lock()
        self._segmenter = segment.Segmenter(labels=labels)
        self._backlog: "queue.Queue" = queue.Queue()
        self._audio: "queue.Queue" = queue.Queue()
        self._max_source = int(max_source_chars)
        self._speaker = speaker
        self._pump: threading.Thread | None = None

        self.base_chars = 0
        """How much of this reply existed before the run started.

        Section 7's speech base offset. Voice speaks only what was newly
        generated for a continuation, so this is not text the segmenter is ever
        given -- it is recorded because "how much of the message was already
        there" is the difference between a turn that spoke a whole reply and one
        that spoke a tail, and a metric that could not tell them apart would be
        misleading about both.
        """

        self.source_chars = 0
        self.sample_rate = 0
        self.error = ""
        self.reason = ""
        self.source_done = False
        self.synthesis_started = False
        self.synthesis_done = False

        self._queued_samples = 0
        self._first_text = 0.0
        self._first_segment = 0.0
        self._first_audio = 0.0
        self._segments_sent = 0
        self._chunks = 0
        self._samples = 0
        self._compute_started = 0.0
        self._compute = 0.0

    # -- text in ----------------------------------------------------------- #

    def add_text(self, delta: str) -> None:
        """More of the reply arrived. Never blocks, never raises.

        Called from inside the Conversation generator, between one visual
        update and the next, which is why it cannot be allowed to wait for
        anything: section 6 is explicit that the LLM generator must not call
        Kokoro, and a blocking append would be the same mistake wearing a
        queue. The most this does is run the segmenter over a few hundred
        characters and put the result on a queue.
        """
        if self.cancelled.is_set() or self.source_done:
            return
        text = str(delta or "")
        if not text:
            return
        try:
            with self._lock:
                if not self._first_text:
                    self._first_text = time.monotonic()
                self.source_chars += len(text)
                if self.source_chars > self._max_source:
                    self._refuse_length()
                    return
                found = self._segmenter.feed(text)
                self._commit(found)
        except Exception:
            logger.debug("Model Chain: a Voice turn could not accept new text", exc_info=True)
            self.cancel("error")

    def complete(self, whole: str = None) -> None:
        """The reply is whole. Flush the tail and stop accepting text.

        ``whole`` is the panel's authoritative cleaned reply where it has one.
        Idempotent: the panel has two paths that reach a finished reply and
        both of them call this.
        """
        if self.source_done:
            return
        try:
            with self._lock:
                self.source_done = True
                found = self._segmenter.flush(whole)
                self._commit(found)
                self._backlog.put_nowait(None)
        except Exception:
            logger.debug("Model Chain: a Voice turn could not be completed", exc_info=True)
            self.cancel("error")

    def _commit(self, found) -> None:
        """Put newly whole segments on the backlog. Holds :attr:`_lock`."""
        for text in found or ():
            if not text.strip():
                continue
            if self._backlog.qsize() >= MAX_BACKLOG_SEGMENTS:
                self._refuse_length()
                return
            if not self._first_segment:
                self._first_segment = time.monotonic()
            self._backlog.put_nowait(text)

    def _refuse_length(self) -> None:
        """Stop the turn at the speech ceiling, visibly. Holds :attr:`_lock`.

        Section 11: the reply itself is untouched and stays on screen. What is
        refused is reading it aloud, and the refusal says so rather than
        speaking the first part of an answer and stopping without a word.
        """
        self.error = ("That reply is longer than Voice Chat reads aloud in one go, so it "
                      "was not spoken. Nothing was cut short without telling you.")
        self.reason = "limit"
        self.source_done = True
        self.cancelled.set()
        self._wake()

    # -- audio out --------------------------------------------------------- #

    def offer_audio(self, pcm: bytes, rate: int, seconds_limit: float = AUDIO_QUEUE_SECONDS
                    ) -> bool:
        """Accept one block of PCM16 for this turn, applying backpressure.

        Called on the runtime's reader thread. Blocks while the queue is full,
        which is the whole point -- the pipe behind it fills, the worker's
        writer stops, and the sherpa callback stops being called. It never
        blocks *past* a cancellation: the wait is a loop around the event, so a
        Stop pressed while the browser has stopped consuming still frees this
        thread immediately.

        Returns False when the turn was cancelled and the block was discarded.
        """
        if self.cancelled.is_set():
            return False
        if not pcm:
            return True
        rate = int(rate or 0) or self.sample_rate or 24000
        samples = len(pcm) // 2
        while not self.cancelled.is_set():
            with self._lock:
                queued = self._queued_samples / float(rate or 1)
                if queued < seconds_limit or self._queued_samples == 0:
                    self.sample_rate = self.sample_rate or rate
                    self._queued_samples += samples
                    self._samples += samples
                    self._chunks += 1
                    if not self._first_audio:
                        self._first_audio = time.monotonic()
                    self._audio.put_nowait(("audio", pcm))
                    self.started.set()
                    return True
            time.sleep(PUMP_POLL)
        return False

    def read_audio(self, timeout: float = PUMP_POLL):
        """One block for the HTTP stream, or ``None`` when there is none yet.

        Raises :class:`StopIteration` semantics through a sentinel rather than
        an exception: the stream iterator wants "more", "not yet" and "over" as
        three ordinary values.
        """
        try:
            kind, payload = self._audio.get(timeout=timeout)
        except queue.Empty:
            return ("wait", b"")
        if kind == "audio":
            with self._lock:
                self._queued_samples = max(0, self._queued_samples - len(payload) // 2)
        return (kind, payload)

    def audio_finished(self) -> None:
        """The worker said this turn's audio is complete."""
        self.synthesis_done = True
        self._audio.put(("end", b""))
        self.finished.set()

    def audio_failed(self, reason: str) -> None:
        self.error = str(reason or "") or "Voice could not read that reply aloud."
        self.reason = self.reason or "error"
        self._audio.put(("end", b""))
        self.finished.set()
        self.cancelled.set()

    # -- cancellation ------------------------------------------------------ #

    def cancel(self, reason: str = "user") -> bool:
        """Stop this turn. Idempotent, never raises, safe from any thread.

        Returns whether this call is the one that cancelled it, which is what
        the ``/cancel`` route reports and what stops two Stop paths from
        logging the same cancellation twice.
        """
        first = not self.cancelled.is_set()
        if first:
            self.reason = self.reason or str(reason or "user")
        self.cancelled.set()
        self._wake()
        return first

    def _wake(self) -> None:
        """Unblock everything waiting on this turn, in both directions."""
        try:
            self._backlog.put_nowait(None)
        except Exception:
            pass
        try:
            self._audio.put_nowait(("end", b""))
        except Exception:
            pass
        self.finished.set()

    def drain_audio(self) -> None:
        """Throw away queued PCM for a turn nobody is listening to.

        Called when the turn is cancelled or its client disappears. Without it,
        a cancelled turn's full queue keeps the runtime's reader thread blocked
        in :meth:`offer_audio` -- section 16's requirement that a full queue
        must never make cancellation wait behind the producer.
        """
        while True:
            try:
                self._audio.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._queued_samples = 0

    # -- the pump ---------------------------------------------------------- #

    def start(self, speaker=None) -> None:
        """Begin synthesising, on a thread of this turn's own.

        A thread rather than a task on the event loop, and a thread per turn
        rather than a pool: there is one active turn, its work is one blocking
        conversation with a subprocess, and a pool would add a queue whose
        cancellation semantics this module would then also own.
        """
        if self._pump is not None:
            return
        self._speaker = speaker or self._speaker
        if self._speaker is None:
            # The runtime by default, imported here rather than at module scope:
            # this module is imported by the runtime's unload path, and a
            # circular import at start-up would cost Conversation its panel.
            import mc_voice_runtime

            self._speaker = mc_voice_runtime
        self._pump = threading.Thread(target=self._run, name="mc-voice-turn", daemon=True)
        self._pump.start()

    def _run(self) -> None:
        """Segments to the worker, in order, until the reply or the user ends it."""
        speaker = self._speaker
        began = False
        try:
            first = self._await_segment(FIRST_SEGMENT_WAIT)
            if first is None:
                self.cancel(self.reason or "empty")
                return
            self.sample_rate = int(speaker.begin_turn(self, self.sid, self.profile) or 0)
            began = True
            self.synthesis_started = True
            self._compute_started = time.monotonic()
            pending = first
            self._watch_for_client()
            while pending is not None and not self.cancelled.is_set():
                speaker.send_segment(self, pending)
                self._segments_sent += 1
                pending = self._await_segment(None)
            if not self.cancelled.is_set():
                speaker.finish_turn(self)
                # The worker answers ``tts_done`` when the last sample is out,
                # which is what sets :attr:`finished`. Waiting for it here is
                # what keeps the turn -- and the browser's "voice is busy" --
                # alive until the audio really is complete.
                while not self.finished.wait(PUMP_POLL):
                    if self.cancelled.is_set():
                        break
        except Exception as exc:
            logger.debug("Model Chain: a Voice turn ended on an error", exc_info=True)
            self.error = self.error or _readable(exc)
            self.reason = self.reason or "error"
            self.cancelled.set()
        finally:
            if self._compute_started:
                self._compute = time.monotonic() - self._compute_started
            if began and self.cancelled.is_set() and not self.synthesis_done:
                try:
                    speaker.cancel_turn(self)
                except Exception:
                    logger.debug("Model Chain: could not tell the worker to stop speaking",
                                 exc_info=True)
            self.finished.set()
            self._audio.put(("end", b""))

    def _watch_for_client(self) -> None:
        """Give up on a turn no browser ever opened. See :attr:`attached`."""

        def watch():
            if not self.attached.wait(CLIENT_WAIT) and self.busy:
                logger.debug("Model Chain: a Voice turn was cancelled — nothing connected "
                             "to its audio stream")
                self.cancel("no client")
                self.drain_audio()

        threading.Thread(target=watch, name="mc-voice-turn-watch", daemon=True).start()

    def _await_segment(self, timeout):
        """The next segment, or ``None`` for "no more".

        ``timeout`` of ``None`` waits for as long as the reply is still being
        written. Both forms wake on cancellation within :data:`PUMP_POLL`.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.cancelled.is_set():
            try:
                found = self._backlog.get(timeout=PUMP_POLL)
            except queue.Empty:
                if deadline is not None and time.monotonic() > deadline:
                    return None
                if self.source_done and self._backlog.empty():
                    return None
                continue
            if found is None:
                if self.source_done and self._backlog.empty():
                    return None
                continue
            return found
        return None

    # -- what may be written down ------------------------------------------ #

    def metrics(self) -> dict:
        """The content-free record of one turn. Section 36's list, exactly.

        Built by naming what may be reported rather than by removing what may
        not: a field added to this class later is absent from a log until
        somebody adds it here on purpose, which is the safer direction for a
        feature whose privacy claim is this specific.
        """
        rate = float(self.sample_rate or 0) or 1.0
        audio_seconds = self._samples / rate
        started = self._first_text or self.created_at
        return {
            "source_chars": self.source_chars,
            "base_chars": self.base_chars,
            "segments": self._segments_sent,
            "chunks": self._chunks,
            "audio_seconds": round(audio_seconds, 2),
            "compute_seconds": round(self._compute, 2),
            "rtf": round(self._compute / audio_seconds, 3) if audio_seconds > 0.01 else None,
            "first_segment_ms": _since(started, self._first_segment),
            "first_audio_ms": _since(started, self._first_audio),
            "cancelled": self.reason,
            "voice_type": "clone" if self.voice_id.startswith("clone:") else "official",
            "sid": self.sid,
        }

    @property
    def busy(self) -> bool:
        """Whether Voice is still working on this turn."""
        return not (self.finished.is_set() or self.cancelled.is_set())


def _since(start: float, mark: float):
    if not start or not mark or mark < start:
        return None
    return int((mark - start) * 1000)


def _readable(exc: BaseException) -> str:
    from mc_voice_runtime import VoiceRuntimeError

    if isinstance(exc, (VoiceRuntimeError, VoiceTurnError)):
        return str(exc)
    return "Voice could not read that reply aloud."


# --------------------------------------------------------------------------- #
# The registry of turns
# --------------------------------------------------------------------------- #


_lock = threading.Lock()
_turns: dict = {}
_active_id = ""
KEEP = 180.0
"""How long a finished turn's identity stays known.

Long enough that a browser reconnecting to a stream it already started finds
the turn rather than a 404, short enough that a day of conversation does not
accumulate one dictionary entry per reply.
"""


def create(voice_id: str = "", sid: int = 0, labels=(), page: str = "",
           speaker=None, profile=None) -> VoiceTurn:
    """Make a turn the active one, cancelling whatever was active before.

    Cancelling the previous turn here rather than leaving it is section 24's
    race written down: a new assistant reply means the last one is over, and a
    turn that is still producing audio when its successor starts is exactly the
    condition where an old reply speaks over a new one.
    """
    global _active_id

    turn = VoiceTurn(voice_id=voice_id, sid=sid, labels=labels, page=page, speaker=speaker,
                     profile=profile)
    previous = None
    with _lock:
        _expire()
        previous = _turns.get(_active_id)
        _turns[turn.id] = turn
        _active_id = turn.id
    if previous is not None and previous.busy:
        previous.cancel("superseded")
        previous.drain_audio()
    return turn


def lookup(token: str):
    with _lock:
        return _turns.get(str(token or ""))


def active():
    with _lock:
        return _turns.get(_active_id)


def cancel(token: str, reason: str = "user") -> bool:
    """Cancel one turn by its opaque id. Idempotent and safe to call for a
    turn that never existed -- the browser sends what it has, and a stale token
    after a WebUI restart is an ordinary thing rather than an error."""
    turn = lookup(token)
    if turn is None:
        return False
    first = turn.cancel(reason)
    turn.drain_audio()
    return first


def cancel_active(reason: str = "user") -> bool:
    """Cancel whatever is speaking now. What Stop and unload call."""
    turn = active()
    if turn is None or not turn.busy:
        return False
    first = turn.cancel(reason)
    turn.drain_audio()
    return first


def busy() -> bool:
    """Whether any turn is still producing or waiting to produce speech."""
    with _lock:
        return any(turn.busy for turn in _turns.values())


def forget_all(reason: str = "shutdown") -> None:
    """Cancel and drop every turn. Used by unload, shutdown and the tests."""
    global _active_id

    with _lock:
        found = list(_turns.values())
        _turns.clear()
        _active_id = ""
    for turn in found:
        try:
            turn.cancel(reason)
            turn.drain_audio()
        except Exception:
            pass


def _expire() -> None:
    """Drop finished turns nobody is going to ask about. Holds :data:`_lock`."""
    now = time.monotonic()
    for key in [key for key, turn in _turns.items()
                if not turn.busy and now - turn.created_at > KEEP and key != _active_id]:
        _turns.pop(key, None)
