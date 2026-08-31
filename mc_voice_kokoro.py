"""Kokoro behind the engine facade: qualified ids in, sherpa SIDs kept inside.

:mod:`mc_voice_registry` is unchanged and stays the source of truth for what
Kokoro voices exist, what they are called and which number each one is. What it
does not do -- and deliberately still does not -- is speak the backend-qualified
dialect the shared code now uses. This module is the adapter between them.

    outside this file   kokoro:official:af_heart
    inside it           official:af_heart, and a numeric sherpa sid

Section 18's boundary, in one sentence: engine-native SID, path and tensor
addresses remain inside the adapter. The turn protocol carries the qualified id,
the parent never sees a number, and ``tests/test_voice_engines.py`` asserts that
``tts_begin`` no longer carries a ``sid`` a shared caller had to know about.

Why a wrapper and not a rewrite
-------------------------------
The registry is the module that found and fixed the V1 bug where every reply was
spoken by Alloy while the manifest claimed Heart. Rewriting it to gain a prefix
would have put that back on the table for a change that adds seven characters to
a string. So the prefix is added and stripped here, the registry keeps its own
spelling and its own tests, and the one thing that changes for it is that
nothing outside asks it for a SID any more.

The delete asterisk, the slot reservation, the bank rebuild transaction and the
"synthesize before you commit" validation are all still the registry's. This
file adds no policy of its own; every function below is a translation.
"""

from __future__ import annotations

import logging

import mc_voice_engines as engines
import mc_voice_registry as registry

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ENGINE = engines.KOKORO

LABEL = "Kokoro"

FAMILY = "kokoro-82m"
"""What the handshake and the telemetry call this backend's model family."""


class KokoroError(RuntimeError):
    """A Kokoro voice operation that could not be completed. Never fatal."""


def _out(entry) -> dict:
    """One registry entry with a qualified id, and without its speaker number.

    The SID is removed rather than renamed. A payload that carried it would be
    a payload a browser could read it from, and section 56 has been explicit
    since V1.1 that a number which did not come through :func:`resolve` is a
    number that can address any speaker in the bank.
    """
    if entry is None:
        return None
    found = {key: value for key, value in dict(entry).items() if key != "sid"}
    found["id"] = engines.qualify(entry["id"], ENGINE)
    found["engine"] = ENGINE
    return found


def entries() -> list:
    """Every Kokoro voice, official first, then custom. Qualified ids."""
    return [_out(entry) for entry in registry.entries()]


def lookup(voice_id: str):
    """One voice by qualified id, or ``None``. An id belonging to another
    engine is ``None`` here rather than an error: the caller is choosing what to
    do about a missing voice, and "it is Sopro's" is not their problem."""
    if not engines.belongs(voice_id, ENGINE):
        return None
    return _out(registry.lookup(engines.native(voice_id)))


def default_id() -> str:
    """The configured Kokoro default, qualified. Always resolves (registry's rule)."""
    return engines.qualify(registry.default_id(), ENGINE)


def default_entry():
    return _out(registry.default_entry())


def set_default(voice_id: str) -> dict:
    """Commit a new Kokoro default. Cannot be given another engine's voice."""
    _mine(voice_id)
    return _out(registry.set_default(engines.native(voice_id)))


def resolve(voice_id: str = "") -> tuple:
    """``(qualified id, entry)`` -- and the sherpa SID *inside* the entry.

    The signature the facade defines, which is deliberately not the registry's
    ``(sid, entry)``. A caller of this is about to open a turn; what it needs is
    something it can put in a message and something it can put on screen, and
    the number belongs to :func:`begin_turn` below and to nothing else.
    """
    chosen = engines.native(voice_id) if engines.belongs(voice_id, ENGINE) else ""
    sid, entry = registry.resolve(chosen)
    found = _out(entry)
    # Carried privately so the runtime can read it without a second registry
    # round trip, and stripped by :mod:`mc_voice_api` before anything is sent
    # to a browser -- see :func:`mc_voice_api._public`.
    found["_sid"] = int(sid)
    # And under the name every engine's entry answers to. ``_handle`` is what
    # the shared turn carries and what the runtime is handed, and it is opaque
    # above this line on purpose: Kokoro's is an integer speaker in a block of
    # floats, Sopro's and Pocket's are their own stable ids, and no shared
    # caller may branch on which (I-10, section 8). ``_sid`` stays beside it
    # because Kokoro's own tests and telemetry name the number.
    found["_handle"] = int(sid)
    return found["id"], found


def rename(voice_id: str, display_name: str) -> dict:
    _mine(voice_id)
    return _out(registry.rename(engines.native(voice_id), display_name))


def delete(voice_id: str) -> dict:
    _mine(voice_id)
    return _out(registry.delete(engines.native(voice_id)))


def warnings() -> list:
    """What is wrong with the Kokoro side that a user can see and act on."""
    try:
        return list(registry.warnings())
    except Exception:
        logger.debug("Model Chain: could not read the Kokoro voice warnings", exc_info=True)
        return []


def capacity() -> dict:
    return registry.capacity()


def test_text() -> str:
    return registry.test_text()


def set_test_text(text: str) -> str:
    return registry.set_test_text(text)


def status():
    """Whether Kokoro can speak, in the shape the engine panel reads.

    A thin pass-through of :func:`mc_voice_models.status` narrowed to TTS,
    because the engine panel asks both engines the same question and Kokoro's
    answer has always been part of a larger object that also covers STT --
    which this engine does not own.
    """
    import mc_voice_models as models

    found = models.status()
    return _Status(ready=bool(found.tts_ready), message=found.tts_message,
                   label=found.tts_label if hasattr(found, "tts_label") else LABEL)


class _Status:
    """The three fields every engine's status answers with.

    A small class rather than a dict so that a caller reading ``.ready`` on the
    wrong object fails where the mistake is, rather than reading ``False`` out
    of a dictionary that never had the key.
    """

    __slots__ = ("ready", "message", "label")

    def __init__(self, ready: bool, message: str, label: str):
        self.ready = bool(ready)
        self.message = str(message or "")
        self.label = str(label or LABEL)


def capabilities() -> dict:
    """What Kokoro can do, as behaviour rather than decoration. Section 8.

    Almost nothing, and that is the honest answer: Kokoro's cloning is a
    separate offline tool with a window of its own rather than a preview
    transaction on this page, it has no engine-global compute settings a user
    may change, and it has no Lab. What it does have is real cancellation, which
    is the one capability the shared Stop path reads.
    """
    return {"clone_preview": False, "rebuild": False, "engine_settings": False,
            "starter_voices": False, "voice_lab": False, "interrupt_mode": "cancel"}


def refusals() -> tuple:
    """The exception types this adapter raises to *refuse* rather than fail.

    Both of them, because a Kokoro voice operation can be refused from either
    side of the boundary: this module refuses an id that is not Kokoro's, and
    the registry underneath refuses a voice that does not exist or a bank that
    has never been built. Declared here so that the facade asks the engine
    rather than deciding that "not Sopro" means the Kokoro registry -- which was
    true while there were two engines and is not a rule that survives a third.
    """
    return (KokoroError, registry.RegistryError)


def public_status() -> dict:
    """Kokoro's operational state, in the shape every engine answers with.

    The common subset section 31 asks for -- installed, ready, message, whether
    a worker is resident, whether the lane is occupied, whether it is draining,
    and what Stop means -- plus ``block``, which is the engine-owned part the
    status payload publishes under this engine's own id.

    ``draining`` is always False here and is present rather than absent on
    purpose: a browser reads one field to decide whether to draw a waiting
    state, and a key that exists on one engine and not another is a key every
    caller has to guess about.
    """
    found = status()
    resident = False
    try:
        import mc_voice_runtime as runtime

        resident = bool((runtime.engine() or {}).get("loaded"))
    except Exception:
        logger.debug("Model Chain: could not read whether the Kokoro worker is resident",
                     exc_info=True)
    import mc_voice_models as models

    whole = models.status()
    return {
        "installed": found.ready,
        # Two readinesses, because Kokoro's engine block has always carried
        # two and they are not the same question. ``ready`` is whether Voice
        # Chat as a whole is set up -- runtime, speech-to-text and
        # text-to-speech -- which is what the browser draws its "not set up"
        # state from; ``tts_ready`` is this engine's own half. Collapsing them
        # made a machine with no Whisper report itself ready.
        "ready": whole.ready,
        "tts_ready": found.ready,
        "message": found.message,
        "worker_resident": resident,
        "engine_busy": False,
        "draining": False,
        "interrupt_mode": "cancel",
        "block": {"installed": found.ready, "message": found.message},
    }


def _mine(voice_id: str) -> None:
    if not engines.belongs(voice_id, ENGINE):
        raise KokoroError("That is not a Kokoro voice.")
